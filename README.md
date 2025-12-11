# Turtle Chi: Robot-Mediated Tai Chi Instruction System
## Jack Archer, Jiwon Moon, Pooja Vegesna
## Final Project Report Sections

---

## 1. Motivation

As the global population ages and demands for geriatric care continue to increase, Tai Chi represents an accessible, meaningful activity, and this project aims to make that instruction more engaging 
and supportive through robotics. Tai Chi is an exercise that involves flowing gentle movements, originated as an ancient martial art in China. We explore a human–robot interaction (HRI) scenario in which a
robot performs the role of a Tai Chi instructor, guiding a user through a predefined sequence of movements. By using computer vision (CV), the robot will observe the human's pose, evaluate performance, 
and provide corrective verbal feedback.

## 2. Prior Research + Identified Gaps

Several studies have explored robot-mediated Tai Chi or dance instruction for adults. Recent vision-based Tai Chi evaluation systems have focused on movement assessment without robotic embodiment. 
Zhao applies deep learning models to evaluate the quality of Tai Chi movements and provide personalized training insights, but it does not incorporate a physical robot capable of demonstrating or 
mirroring motions [5]. Han et. al constructs a labeled dataset of expert and novice demonstrations and uses pose-based features to score performance; however, this system performs only offline evaluation 
and does not close the loop by providing interactive feedback or adapting a robot's behavior based on user performance [2]. In HRI, Zheng et. al explored how an anthropomorphic robot (Nao) can 
successfully lead users through pre-scripted Tai Chi sequences, yet did not provide evaluation to user's performance, limiting the interaction to an one-way demonstration [4]. Also, Granados et. al 
explored physical HRI in which a robot guides a defined dance training, but it didn't provide any explicit feedback [1]. For non-Tai Chi movements, Olikkal et. al explored real-time hand-gesture 
imitation studies, but only focused on upper-limb gestures and did not address full-body pose estimation, which is essential for Tai Chi [3].

Overall, to our knowledge, no existing system integrates real-time full-body pose evaluation, physical robotic demonstration, and adaptive verbal feedback in a physical movement (especially Tai Chi) 
teaching context. This project aims to address this gap to explore how robots can provide engaging learning experiences.

---

## System Architecture

The Turtle Chi system implements a closed-loop human-robot interaction pipeline for Tai Chi instruction, combining robotic demonstration, real-time pose evaluation, and adaptive feedback.
The architecture consists of three major components that work in concert to create an engaging learning experience.

### 1. Movement Control & Orchestration Node (`interaction_node.py`)

This ROS node serves as the central controller, managing the entire teaching session workflow. The key components include:

**Movement Sequence Management:**
The node stores three distinct Tai Chi movements as sequences of joint positions for the TurtleBot4's OpenManipulator arm. All three movements use keyframe interpolation to generate smooth transitions. 
The `_interpolate_poses()` method creates intermediate poses between keyframes using linear interpolation:

- **Movement 1**: 8 keyframes with 2 interpolation steps between each, resulting in 22 total poses
- **Movement 2**: 5 keyframes with 2 interpolation steps between each, resulting in 13 total poses  
- **Movement 3**: 8 keyframes with 4 interpolation steps between each, resulting in 36 total poses

This approach balances motion smoothness with computational efficiency, with Movement 3's higher interpolation density (4 steps) accommodating more complex joint rotations and longer trajectory segments.

**Robot Control Interface:**
- **Arm Control**: The `move_arm()` method publishes `JointTrajectory` messages with 4 joint positions (joint1-4) and configurable duration parameters, allowing precise control over movement speed. There was no particular reason to use the RoS2 provided controller message (trajectory_msgs/JointTrajectory.msg)aside from the fact that we were looking at RoS2 documentation while coding this.
- **Gripper Animation**: The `talking_animation()` method creates a "talking" effect by rapidly opening and closing the gripper during audio playback, enhancing engagement
- **Base Movement**: The `celebration_spin()` method commands the robot base to perform a 360&deg; rotation at 0.8 rad/s when the user successfully completes a pose, using `Twist` messages on the `/cmd_vel` topic

**Audio-Visual Feedback System:**
The node integrates synchronized audio playback using the `aplay` command-line tool, with both blocking and non-blocking modes. Audio files include welcome messages, movement-specific instructions, and performance feedback. The system coordinates audio with robot animations to create a cohesive teaching experience.

**Teaching Loop Implementation:**
The `teach_movement()` method implements a structured four-step pedagogical sequence:
1. **Silent Demonstration**: Robot performs the movement while playing introductory audio
2. **Guided Practice**: Robot repeats the movement, allowing the user to follow along
3. **Hold for Evaluation**: User maintains final pose while system captures and evaluates
4. **Adaptive Feedback**: Based on evaluation results, provides encouragement (celebration spin) or corrective guidance

**Multi-Model Evaluation Strategy:**
For Movement 1, the system implements a two-tier evaluation approach. If the primary classifier returns "incorrect," it switches to a secondary model (model 1 low) that distinguishes between "arms are low" and "general incorrect". This provides more nuanced feedback, directing users to specific corrections (such as "lower your arms" in this case) rather than generic "try again" messages.

### 2. Pose Estimation & Classification Node (`pose_node.py`)

This node handles real-time computer vision processing and pose evaluation using a pipeline that combines state-of-the-art pose estimation with custom-trained classifiers.

**Pose Detection Pipeline:**
The system uses TensorFlow Hub's MoveNet SinglePose Lightning model, a pre-trained neural network optimized for real-time human pose estimation. The pipeline:
1. Subscribes to camera images from the TurtleBot4's OAK-D camera (`/tbXX/oakd/rgb/preview/image_raw`)
2. Resizes and preprocesses images to 192×192 pixels with padding to maintain aspect ratio
3. Extracts 17 keypoints (COCO format) with (y, x, confidence) coordinates
4. Validates detection quality using average confidence threshold (0.3 minimum)

**Feature Engineering:**
The `extract_features()` function transforms raw keypoints into a 43-dimensional feature vector optimized for Tai Chi pose discrimination:
- **Normalization** (via `normalize_keypoints()`): Centers pose on torso midpoint and scales by torso dimensions (shoulder width + torso length), making the system invariant to camera distance and user size
- **Geometric Features**: Calculates 4 joint angles (left/right elbow, left/right shoulder) using the `angle()` function to capture arm configuration
- **Spatial Features**: Computes torso tilt (x, y components), arm heights relative to shoulders, and feet distance to capture overall body positioning
- **Raw Coordinates**: Includes all 17 normalized keypoint positions (34 values) to preserve spatial relationships

This feature set was designed through iterative experimentation to capture the essential geometric properties that distinguish correct from incorrect Tai Chi poses.

**Multi-Model Classification System:**
The node loads and manages 5 distinct MLP (Multi-Layer Perceptron) classifiers at startup:
- **Models 1-4**: Movement-specific binary classifiers trained to evaluate correctness of each Tai Chi movement
- **Model 1 Low**: Secondary quality classifier for Movement 1, distinguishing "low arms" from "generalized incorrect"

Model selection occurs dynamically via the `/select_movement` topic, allowing the orchestration node to switch evaluators as the teaching session progresses. Each model maintains its own scaler parameters (mean, std) loaded from `.npz` files, ensuring features are normalized identically to training conditions.

**Classification Architecture:**
Each MLP uses a 43 -> 64 -> 32 -> 1 architecture with ReLU activation in hidden layers and sigmoid output. The models were trained from scratch (not using scikit-learn) with weight initialization (Xavier/He) and binary cross-entropy loss. This custom implementation provides full control over the classification pipeline and eliminates external dependencies in the ROS node.

### 3. Training & Validation Pipeline (`train_mlp_classifier.py`, `test_mlp_classifier.py`)

**Data Collection & Annotation:**
The training data consists of JSON files containing MoveNet-extracted keypoints from demonstration videos. Each movement has separate `correct.json` and `incorrect.json` files in the `dataset/` directory, with keypoints stored as 17×3 arrays (y, x, confidence). This dataset was collected by recording multiple users attempting each Tai Chi movement and manually labeling demonstrations.

**MLP Implementation Details:**
The `MLPBinaryClassifier` class implements a fully-connected neural network entirely in NumPy:
- Forward pass computes layer-wise transformations: z₁ = X·W₁ + b₁, a₁ = ReLU(z₁), etc.
- Backward pass implements manual gradient computation through backpropagation
- Weight updates use vanilla gradient descent (no momentum or Adam)
- Training includes epoch-wise shuffling to prevent ordering bias

**Training Process:**
1. Loads and concatenates correct (label=1) and incorrect (label=0) samples
2. Extracts 43 features from each keypoint set
3. Computes dataset statistics (mean, std) and saves as scaler
4. Trains for 600 epochs with learning rate 5e-3
5. Saves model weights (W₁, b₁, W₂, b₂, W₃, b₃) to `.npz` file

The training script reports loss and accuracy every 100 epochs. The models are stored in `models/ver3/`. The accuracy for each model is as follows: (1) movement_1: 100% (2) movement_2: 97.8% (3) 97.8%.

### Integration & Information Flow

The complete system operates through publish-subscribe coordination:

1. **Session Initialization**: Orchestration node loads movement sequences and audio files
2. **Movement Demonstration**: Orchestration node publishes arm trajectories, robot performs movement
3. **Evaluation Trigger**: After demonstration, orchestration publishes `True` on `/trigger_capture`
4. **Pose Processing**: Pose node captures image, runs MoveNet, extracts features, classifies with current model
5. **Result Publishing**: Pose node publishes result string ("correct", "incorrect", "no_person") on `/pose_result`
6. **Feedback Execution**: Orchestration node reads result, executes celebration/correction behavior
7. **Iteration**: Process repeats for each movement in the session

This architecture enables real-time interaction (<2s evaluation latency) while maintaining modularity—each node can be developed, tested, and debugged independently.

---

## ROS Node Diagram

The system consists of two primary ROS nodes with the following communication structure:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TurtleBot4 Robot Platform                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Hardware Interfaces                                          │  │
│  │  - OpenManipulator Arm (4-DOF)                                │  │
│  │  - Gripper Actuators                                          │  │
│  │  - Mobile Base (Differential Drive)                           │  │
│  │  - OAK-D Stereo Camera                                        │  │
│  │  - Audio System (No Machine speakers)                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    ↕
        ┌────────────────────────────────────────────────────┐
        │                                                    │
        ↓                                                    ↓

┌──────────────────────────────┐          ┌──────────────────────────────────┐
│  THREE_MOVEMENT_TAI_CHI_NODE │          │ MULTI_MODEL_TAI_CHI_POSE_NODE    │
│  (Orchestration & Control)   │          │ (Vision & Evaluation)            │
├──────────────────────────────┤          ├──────────────────────────────────┤
│                              │          │                                  │
│ PUBLISHES:                   │          │ SUBSCRIBES:                      │
│  - /trigger_capture (Bool)   │────────→ │  - /trigger_capture (Bool)       │
│  - /select_movement (Int32)  │────────→ │  - /select_movement (Int32)      │
│  - /arm_controller/          │          │  - /tbXX/oakd/rgb/preview/       │
│    joint_trajectory          │          │    image_raw (Image)             │
│    (JointTrajectory)         │          │                                  │
│  - /tbXX/target_gripper_     │          │ PUBLISHES:                       │
│    position                  │          │  • /pose_result (String)         │
│    (ArmGripperPosition)      │          │                                  │
│  - /tbXX/cmd_vel (Twist)     │          │ DISPLAYS:                        │
│                              │          │  • OpenCV window with live feed  │
│ SUBSCRIBES:                  │          │                                  │
│  - /pose_result (String)     │←──────── │                                  │
│                              │          │ COMPONENTS:                      │
│ COMPONENTS:                  │          │  • MoveNet Pose Estimator        │
│  • Movement Sequences        │          │  • 5× MLP Classifiers            │
│  • Audio Player (aplay)      │          │  • Feature Extractor             │
│  • Teaching Logic            │          │  • Normalization Engine          │
│  • Celebration Behaviors     │          │                                  │
│                              │          │ DATA:                            │
│ DATA:                        │          │  models/ver3/                    │
│  • 3 Tai Chi movement arrays │          │   • movement_1_mlp_*.npz         │
│  • Audio files (WAV)         │          │   • movement_2_mlp_*.npz         │
│  • Neutral pose config       │          │   • movement_3_mlp_*.npz         │
│                              │          │   • movement_4_mlp_*.npz         │
│                              │          │   • movement_1_low_mlp.npz       │
│                              │          │   • *_scaler_*.npz (5 files)     │
└──────────────────────────────┘          └──────────────────────────────────┘

                    MESSAGE FLOW SEQUENCE:

1. [Orchestration] Publishes arm trajectory → Robot performs movement
2. [Orchestration] Publishes Int32 on /select_movement → [Pose] Switches model
3. [Orchestration] Publishes True on /trigger_capture → [Pose] Captures frame
4. [Pose] Processes image with MoveNet → Extracts 17 keypoints
5. [Pose] Computes 43 features → Classifies with selected MLP
6. [Pose] Publishes result ("correct"/"incorrect") on /pose_result
7. [Orchestration] Receives result → Executes celebration or correction
8. [Orchestration] Returns arm to neutral → Proceeds to next movement

                     SUMMARY OF TOPICS USED:

┌─────────────────────────┬──────────────────┬───────────────┬─────────────────┐
│ Topic Name              │ Message Type     │ Publisher     │ Subscriber      │
├─────────────────────────┼──────────────────┼───────────────┼─────────────────┤
│ /trigger_capture        │ std_msgs/Bool    │ Orchestration │ Pose            │
│ /select_movement        │ std_msgs/Int32   │ Orchestration │ Pose            │
│ /pose_result            │ std_msgs/String  │ Pose          │ Orchestration   │
│ /arm_controller/        │ trajectory_msgs/ │ Orchestration │ Arm Controller  │
│   joint_trajectory      │   JointTrajectory│               │                 │
│ /tb11/target_gripper_   │ omx_cpp_interface│ Orchestration │ Gripper Driver  │
│   position              │   /ArmGripper... │               │                 │
│ /tb11/cmd_vel           │ geometry_msgs/   │ Orchestration │ Base Controller │
│                         │   Twist          │               │                 │
│ /tb11/oakd/rgb/preview/ │ sensor_msgs/     │ Camera Driver │ Pose            │
│   image_raw             │   Image          │               │                 │
└─────────────────────────┴──────────────────┴───────────────┴─────────────────┘
```

**Key Design Decisions:**
- **Unidirectional trigger flow**: Orchestration always initiates evaluation, preventing race conditions
- **Model hot-swapping**: All 5 models loaded at startup. Switching via topic publish is instantaneous.
- **Loose coupling**: Nodes communicate only through ROS topics, enabling independent development
- **Synchronous evaluation**: Orchestration blocks waiting for pose_result (5s timeout)

---

## How to run:

### Prerequisites
Ensure the following are installed on your TurtleBot4:
- ROS 2 (Humble)
- Python 3.10+
- Required Python packages: `rclpy`, `opencv-python`, `cv-bridge`, `numpy`, `tensorflow`, `tensorflow-hub`
- Audio playback utility: `aplay` (usually pre-installed on Ubuntu)

### Setup

1. **Clone/Copy Project Files**
   ```bash
   cd ~/intro_robo_ws/src/
   # Make sure turtle_chi package exists with the following structure:
   # turtle_chi/
   # ├── turtle_chi/
   # │   ├── three_movement_tai_chi_node.py
   # │   ├── multi_model_tai_chi_pose_node.py
   # │   ├── *.WAV (audio files)
   # ├── models/
   # │   └── ver3/
   # │       ├── movement_1_mlp_scratch_ver3.npz
   # │       ├── movement_1_scaler_scratch_ver3.npz
   # │       ├── ... (other model files)
   # ├── package.xml
   # ├── setup.py
   ```

2. **Build the Workspace**
   ```bash
   cd ~/intro_robo_ws
   colcon build --packages-select turtle_chi
   source install/setup.bash
   ```

### Running the System


The robot will immediately begin the teaching session.

### Adjusting Parameters

**Disable Audio** (for testing/debugging):
```bash
ros2 run turtle_chi three_movement_tai_chi_node \
  --ros-args -p use_audio:=False
```

**Change Movement Speed**:
```bash
# Faster movements (1.0 second per pose)
ros2 run turtle_chi three_movement_tai_chi_node \
  --ros-args -p step_duration:=1.0

# Slower movements (2.5 seconds per pose)
ros2 run turtle_chi three_movement_tai_chi_node \
  --ros-args -p step_duration:=2.5
```

**Adjust Pose Detection Sensitivity**:
```bash
# More strict detection (only high-confidence keypoints)
ros2 run turtle_chi multi_model_tai_chi_pose_node \
  --ros-args -p min_keypoint_confidence:=0.5

# More lenient detection
ros2 run turtle_chi multi_model_tai_chi_pose_node \
  --ros-args -p min_keypoint_confidence:=0.2
```

### Testing Individual Components

**Test Movement Execution Only** (modify `three_movement_tai_chi_node.py`):
```python
# In main() function, comment out:
# node.run_full_session()

# Uncomment one of:
node.teach_movement(1)  # Test Movement 1 only
# node.teach_movement(2)  # Test Movement 2 only
# node.teach_movement(3)  # Test Movement 3 only
# node.celebration_spin()  # Test celebration only
```

**Test Pose Estimation Only**:
```bash
# Terminal 1: Run pose node
ros2 run turtle_chi multi_model_tai_chi_pose_node

# Terminal 2: Manually trigger evaluation
ros2 topic pub --once /select_movement std_msgs/Int32 "{data: 1}"
ros2 topic pub --once /trigger_capture std_msgs/Bool "{data: true}"

# Terminal 3: View result
ros2 topic echo /pose_result
```

### Expected Session Flow

When running successfully, you should observe:

1. **Welcome Phase** (~10s)
   - Robot greets user with audio
   - Gripper performs "talking" animation
   - Arm remains in neutral position

2. **Movement 1 Teaching** (~45s)
   - Robot demonstrates 11-pose sequence
   - User follows along
   - Robot evaluates final pose
   - Celebration spin if correct
   - If incorrect, run additional classification model.
     - If arms are low, provide corrective feedback and start Movement 1 Teaching again.

3. **Movement 2 Teaching** (~50s)
   - Similar structure with 12-pose sequence
   - If incorrect, run additional classification model.
     - If legs are too far apart, provide corrective feedback and start Movement 2 Teaching again.
   - Includes complex rotations

4. **Movement 3 Teaching** (~90s)
   - Extended 28-pose sequence
   - No additional corrective feedback as Movement 1 or 2.
   - Multiple repetitions of key poses

5. **Session Complete**
   - Final message
   - Robot returns to neutral

**Total session duration: ~4-5 minutes**

### Safety Notes

- Ensure atleast 2m clearance around robot during celebration spin
- Keep hands away from gripper during animations
- Emergency stop: `Ctrl+C` in any terminal
- If arm moves erratically, kill nodes and restart from neutral position
