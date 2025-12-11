#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32, Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from omx_cpp_interface.msg import ArmGripperPosition
from geometry_msgs.msg import Twist
import time
import os
import subprocess
import numpy as np


class ThreeMovementTaiChiNode(Node):
    def __init__(self):
        super().__init__('tai_chi_interaction')
        
        self.declare_parameter('trigger_topic', '/trigger_capture')
        self.declare_parameter('result_topic', '/pose_result')
        self.declare_parameter('model_select_topic', '/select_movement')
        self.declare_parameter('arm_topic', '/arm_controller/joint_trajectory')
        self.declare_parameter('gripper_topic', '/tb11/target_gripper_position')
        self.declare_parameter('cmd_vel_topic', '/tb11/cmd_vel')
        self.declare_parameter('audio_dir', '/home/jarch/intro_robo_ws/src/turtle_chi/turtle_chi/')
        self.declare_parameter('use_audio', True)
        self.declare_parameter('step_duration', 1.8)
        
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.model_select_topic = self.get_parameter('model_select_topic').value
        self.arm_topic = self.get_parameter('arm_topic').value
        self.gripper_topic = self.get_parameter('gripper_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.audio_dir = self.get_parameter('audio_dir').value
        self.use_audio = self.get_parameter('use_audio').value
        self.step_duration = self.get_parameter('step_duration').value
        
        self.get_logger().info("="*60)
        self.get_logger().info("Enhanced Three-Movement Tai Chi Node")
        self.get_logger().info("="*60)
        
        # Publishers
        self.trigger_pub = self.create_publisher(Bool, self.trigger_topic, 10)
        self.arm_pub = self.create_publisher(JointTrajectory, self.arm_topic, 10)
        self.model_select_pub = self.create_publisher(Int32, self.model_select_topic, 10)
        self.gripper_pub = self.create_publisher(ArmGripperPosition, self.gripper_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # Subscriber
        self.result_sub = self.create_subscription(
            String,
            self.result_topic,
            self.result_callback,
            10
        )
        
        self.latest_result = None
        self.waiting_for_result = False

        # Movement 1 
        self.movement_1_poses = [
            [0.0, -0.023, 0.003, 0.0],
            [0.0, 1.499, 0.209, 0.0],
            [0.0, 1.496, -0.407, 0.0], 
            [0.0, 1.496, -0.407, 0.0],
            [0.0, 1.116, -0.699, 0.0],
            [0.0, 0.545, -0.897, 0.0],
            [0.0, 0.545, -1.197, 0.0],
            [0.0, 0.545, -1.197, 0.0],
            [0.0, 0.003, -1.197, 0.0],
            [0.0, -0.402, -1.197, 0.599],
            [0.0, -0.402, -1.197, 0.0],
        ]
 
        self.movement_2_poses = self._create_smooth_movement_2()
 
        self.movement_3_poses = self._create_smooth_movement_3()

        self.movements = {
            1: {
                'name': 'Movement 1 : Flow Sequence',
                'poses': self.movement_1_poses,
            },
            2: {
                'name': 'Movement 2 : Rotation Return',
                'poses': self.movement_2_poses,
            },
            3: {
                'name': 'Movement 3 : Extended Cycle',
                'poses': self.movement_3_poses,
            }
        }

        #Pose first audio files
        self.pose_first_audio = ["Pose_1_audio.WAV", "Pose_2_audio.WAV", "Pose_3_audio.WAV"]

        # Audio files
        self.audio_files = {
            "welcome": "Welcome.wav",
            "intro": "intro.wav",
            "good_job": "good job ur pose worked.wav",
            "stay_for_evaluate": "stay_for_evaluate.wav",
            "try_again": "Jacks_Mom.wav",
            "low_arms": "low_arms.wav", 
        }

        self.neutral_pose = [0.0, -0.023, 0.003, 0.0]

        self.get_logger().info("="*60)
        self.get_logger().info("Ready! 3 Movements Loaded:")
        self.get_logger().info(f"  Movement 1: {len(self.movement_1_poses)} poses")
        self.get_logger().info(f"  Movement 2: {len(self.movement_2_poses)} poses")
        self.get_logger().info(f"  Movement 3: {len(self.movement_3_poses)} poses )")
        self.get_logger().info(f"  Step duration: {self.step_duration}s")
        self.get_logger().info("="*60)
    
    def _interpolate_poses(self, start_pose, end_pose, num_steps=3): 
        interpolated = []
        for i in range(1, num_steps + 1):
            alpha = i / (num_steps + 1)
            interp_pose = [
                start_pose[j] + alpha * (end_pose[j] - start_pose[j])
                for j in range(4)
            ]
            interpolated.append(interp_pose)
        return interpolated
    
    def _create_smooth_movement_2(self): 
        key_poses = [
            [-0.008, -1.005, 0.715, 0.302],   # Start
            [1.571, -1.950, -2.0, 0.0],        # Large rotation
            [1.571, 0.0, -2.0, 0.0],           # Arms extended
            [1.571, -1.300, 1.150, -1.4],      # Complex position
        ]
        
        smooth_poses = []
        for i in range(len(key_poses)):
            smooth_poses.append(key_poses[i])
            if i < len(key_poses) - 1:
                # Add 2 interpolation poses between each key pose
                interpolated = self._interpolate_poses(key_poses[i], key_poses[i+1], num_steps=2)
                smooth_poses.extend(interpolated)
        
        return smooth_poses
    
    def _create_smooth_movement_3(self): 
        key_poses = [
            [-0.008, -1.005, 0.715, 0.302],    # Start
            [1.571, 1.950, -2.0, 0.0],          # Opposite rotation
            [1.571, 0.0, -2.0, 0.0],            # Extended
            [1.571, -1.300, 1.050, 0.0],        # Transition
            [1.57, 1.249, -0.451, -0.499],      # Complex pose
            [1.571, 0.0, -2.0, 0.0],            # Extended again
            [1.571, -1.300, 1.050, 0.0],        # Repeat transition
            [1.57, 1.249, -0.451, -0.499],      # Repeat complex
            [-0.008, -1.005, 0.715, 0.302],    # End
        ]
        
        smooth_poses = []
        for i in range(len(key_poses)):
            smooth_poses.append(key_poses[i])
            if i < len(key_poses) - 1:
                # Add 1 interpolation pose between each ? maybe 2? 
                interpolated = self._interpolate_poses(key_poses[i], key_poses[i+1], num_steps=3)
                smooth_poses.extend(interpolated)
        
        return smooth_poses
    
    def play_audio(self, filename, blocking=True):
        if not self.use_audio:
            self.get_logger().info(f"[Audio: {filename}]")
            return None
        
        audio_path = os.path.join(self.audio_dir, filename)
        
        if not os.path.exists(audio_path):
            self.get_logger().warn(f"Audio file not found: {audio_path}")
            return None

        try:
            self.get_logger().info(f" {filename}")
            
            if blocking:
                subprocess.run(['aplay', '-q', audio_path], check=True)
                return None
            else:
                process = subprocess.Popen(['aplay', '-q', audio_path])
                return process
                
        except Exception as e:
            self.get_logger().error(f"Error playing audio: {e}")
            return None
    
    def move_arm(self, joint_positions, duration_sec=None):
        if duration_sec is None:
            duration_sec = self.step_duration
            
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(
            sec=int(duration_sec), 
            nanosec=int((duration_sec % 1) * 1e9)
        )
        
        msg.points = [point]
        self.arm_pub.publish(msg)
    
    def move_gripper(self, position): 
        msg = ArmGripperPosition(left_gripper=position,right_gripper=position)
        self.gripper_pub.publish(msg)
    
    def talking_animation(self, duration=5.0): 
        self.get_logger().info("  Robot talking...")
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            self.move_gripper(0.01)  # Open
            time.sleep(0.3)
            self.move_gripper(-0.01)  # Close
            time.sleep(0.3)
        
        self.move_gripper(0.01)  # End open

    def celebration_spin(self): 
        self.get_logger().info("  Celebration spin!")
        
        # Calculate spin parameters
        angular_velocity = 0.8  # rad/s
        spin_duration = (2*3.1459) / angular_velocity
        # Start spinning
        twist = Twist()
        twist.angular.z = angular_velocity
        
        start_time = time.time()

        while (time.time() - start_time) < spin_duration:
            self.cmd_vel_pub.publish(twist)
            
            # Open/close gripper while spinning
            # self.move_gripper(1.0)
            # time.sleep(0.8)
            # self.move_gripper(0.0)
            # time.sleep(0.8)
        
        # Stop spinning
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.move_gripper(0.0)
    
    def result_callback(self, msg):
        self.latest_result = msg.data
        if self.waiting_for_result:
            self.get_logger().info(f" Result: {msg.data}")

    def perform_movement_sequence(self, movement_num, play_audio_cues=False):
        """Perform a movement's pose sequence"""
        movement = self.movements[movement_num]
        poses = movement['poses']
        # audios = movement['audios']

        #if play_audio_cues:
            #self.play_audio(audios[0], blocking = False)
        # audio_interval = len(poses) // len(audios) if play_audio_cues else 0
        # audio_idx = 0

        for i, pose in enumerate(poses): 
        #     if play_audio_cues and audio_interval > 0 and i % audio_interval == 0 and audio_idx < len(audios):
        #         self.play_audio(audios[audio_idx], blocking=False)
        #         audio_idx += 1
            
            # Move arm
            self.move_arm(pose)
            time.sleep(self.step_duration + 0.1)

    def evaluate_with_secondary_model(self):
        self.get_logger().info("\n  Secondary evaluation (low vs incorrect)...")
        
        # Switch to model 5 (secondary quality classifier)
        model_msg = Int32()
        model_msg.data = 5
        self.model_select_pub.publish(model_msg)
        time.sleep(0.5)
        
        # Trigger evaluation
        self.latest_result = None
        self.waiting_for_result = True
        
        trigger_msg = Bool()
        trigger_msg.data = True
        self.trigger_pub.publish(trigger_msg)
        
        # Wait for result
        wait_start = time.time()
        while (time.time() - wait_start) < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_result is not None:
                break
        
        self.waiting_for_result = False
        return self.latest_result

    def teach_movement(self, movement_num):
        """Teach a single movement"""
        movement = self.movements[movement_num]
        
        self.get_logger().info("\n" + "─"*60)
        self.get_logger().info(f" Teaching {movement['name']}")
        self.get_logger().info("─"*60)
        
        # Step 1: Silent demonstration
        self.get_logger().info("\n Watch the demonstration")
        self.play_audio(self.pose_first_audio[movement_num-1], blocking = False)
        self.perform_movement_sequence(movement_num, play_audio_cues=False)
        
        self.get_logger().info("  Holding final pose...")
        time.sleep(2.0)
        
        # Return to neutral
        self.get_logger().info("  Returning to start...")
        self.move_arm(self.neutral_pose, duration_sec=1.5)
        time.sleep(2.0)
        
        # Step 2: User's turn with audio cues
        self.get_logger().info("\n Your turn - Follow along!")
        time.sleep(1.0)
        
        self.perform_movement_sequence(movement_num, play_audio_cues=False)

        # Step 3: Hold for evaluation
        self.get_logger().info("\n Hold your final pose!")
        self.play_audio(self.audio_files["stay_for_evaluate"])
        time.sleep(1.0)

        # Step 4: Switch model and evaluate
        self.get_logger().info(f"\n Switching to Movement {movement_num} model...")
        model_msg = Int32()
        model_msg.data = movement_num
        self.model_select_pub.publish(model_msg)
        time.sleep(0.5)

        self.get_logger().info(f" Evaluating {movement['name']}...")
        self.latest_result = None
        self.waiting_for_result = True

        trigger_msg = Bool()
        trigger_msg.data = True
        self.trigger_pub.publish(trigger_msg)
        
        # Wait for result
        wait_start = time.time()
        while (time.time() - wait_start) < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_result is not None:
                break
        
        self.waiting_for_result = False
        
        # Step 5: Feedback
        self.get_logger().info("\n Feedback:")
        
        if self.latest_result == "correct":
            self.get_logger().info(f" {movement['name']} CORRECT!")
            
            # Play audio non-blocking
            audio_process = self.play_audio(self.audio_files["good_job"], blocking=False)
             
            self.celebration_spin()
            
            # Wait for audio to finish
            if audio_process:
                audio_process.wait()
                
        elif self.latest_result == "incorrect":
            if movement_num == 1:
                self.get_logger().info(f" {movement['name']} incorrect, checking quality level...")
                
                secondary_result = self.evaluate_with_secondary_model()
                
                if secondary_result == "correct":
                    self.get_logger().info("  Quality: LOW : Almost there!")
                    self.play_audio(self.audio_files["low_arms"])
                else:
                    self.get_logger().info("  Quality: INCORRECT : Try again")
                    self.play_audio(self.audio_files["try_again"])
            else:
                self.get_logger().info(f" {movement['name']} needs improvement")
                self.play_audio(self.audio_files["try_again"])
                
        elif self.latest_result == "no_person":
            self.get_logger().warn(" No person detected")
        else:
            self.get_logger().warn(f" Unexpected result: {self.latest_result}")
        
        time.sleep(2.0)
        
        # Return to neutral
        self.get_logger().info("\n Returning to neutral...")
        self.move_arm(self.neutral_pose, duration_sec=1.5)
        time.sleep(2.0)

    def run_full_session(self):
        """Run complete 3-movement teaching session"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info(" Starting 3-Movement Tai Chi Session!")
        self.get_logger().info("="*60)
        
        self.get_logger().info("\n Welcome to Tai Chi with Turtle Chi!")
        audio_process = self.play_audio(self.audio_files["welcome"], blocking=False)
        
        self.talking_animation(duration=5.0)
         
        if audio_process:
            audio_process.wait()
        
        time.sleep(1.0)
        
        # Teach movements 1-3
        for movement_num in range(1, 4):
            self.teach_movement(movement_num)
            time.sleep(5)

        # Session complete
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info(" Session Complete!")
        self.get_logger().info("    You've learned 3 Tai Chi movements!")
        self.get_logger().info("="*60)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ThreeMovementTaiChiNode()
        
        time.sleep(2.0)
        
        # Run full session 
        node.run_full_session()
        # node.celebration_spin()         
        # node.teach_movement(1)
        # node.teach_movement(2)
        # node.teach_movement(3)
        
        time.sleep(2.0)
        
    except KeyboardInterrupt:
        print("\n Shutting down...")
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
