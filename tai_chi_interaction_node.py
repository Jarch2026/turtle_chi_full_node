#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import os
import subprocess

class ThreeMovementTaiChiNode(Node):
    def __init__(self):
        super().__init__('tai_chi_interaction')
        
        self.declare_parameter('trigger_topic', '/trigger_capture')
        self.declare_parameter('result_topic', '/pose_result')
        self.declare_parameter('model_select_topic', '/select_movement')
        self.declare_parameter('arm_topic', '/arm_controller/joint_trajectory')
        self.declare_parameter('audio_dir', '/home/poojavegesna/intro_robo_ws/src/turtle_chi/turtle_chi/')
        self.declare_parameter('use_audio', True)
        self.declare_parameter('step_duration', 0.8)
        
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.model_select_topic = self.get_parameter('model_select_topic').value
        self.arm_topic = self.get_parameter('arm_topic').value
        self.audio_dir = self.get_parameter('audio_dir').value
        self.use_audio = self.get_parameter('use_audio').value
        self.step_duration = self.get_parameter('step_duration').value
        
        self.get_logger().info("="*60)
        self.get_logger().info("Three-Movement Tai Chi Interaction Node")
        self.get_logger().info("="*60)
        
        # Publishers
        self.trigger_pub = self.create_publisher(Bool, self.trigger_topic, 10)
        self.arm_pub = self.create_publisher(JointTrajectory, self.arm_topic, 10)
        self.model_select_pub = self.create_publisher(Int32, self.model_select_topic, 10)
        
        # Subscriber
        self.result_sub = self.create_subscription(
            String,
            self.result_topic,
            self.result_callback,
            10
        )
        
        self.latest_result = None
        self.waiting_for_result = False

        # Movement 1: Flow Sequence (11 poses)
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

        # Movement 2: Rotation Return (5 poses)
        self.movement_2_poses = [
            [-0.008, -1.005, 0.715, 0.302],
            [1.571, -1.950, -2.0, 0.0],
            [1.571, 0.0, -2.0, 0.0],
            [1.571, -1.300, 1.150, -1.4],
        ]

        # Movement 3: Extended Cycle (9 poses)
        self.movement_3_poses = [
            [-0.008, -1.005, 0.715, 0.302],
            [1.571, 1.950, -2.0, 0.0],
            [1.571, 0.0, -2.0, 0.0],
            [1.571, -1.300, 1.050, 0.0],
            [1.57, 1.249, -0.451, -0.499],
            [1.571, 0.0, -2.0, 0.0],
            [1.571, -1.300, 1.050, 0.0],
            [1.57, 1.249, -0.451, -0.499],
            [-0.008, -1.005, 0.715, 0.302],
        ]

        self.movements = {
            1: {
                'name': 'Movement 1 : Flow Sequence',
                'poses': self.movement_1_poses,
                'audios': ["downward_flow.wav", "upward_flow.wav", "back_to_center.wav"]
            },
            2: {
                'name': 'Movement 2 : Rotation Return',
                'poses': self.movement_2_poses,
                'audios': ["downward_flow.wav", "upward_flow.wav", "back_to_center.wav"]
            },
            3: {
                'name': 'Movement 3 : Extended Cycle',
                'poses': self.movement_3_poses,
                'audios': ["downward_flow.wav", "upward_flow.wav", "back_to_center.wav"]
            }
        }

        # General audio files
        self.audio_files = {
            "intro": "intro.wav",
            "good_job": "good job ur pose worked.wav",
            "stay_for_evaluate": "stay_for_evaluate.wav",
            "try_again": "Jacks_Mom.wav",
        }

        self.neutral_pose = [0.0, -0.023, 0.003, 0.0]

        self.get_logger().info("="*60)
        self.get_logger().info("Ready! 3 Movements Loaded:")
        self.get_logger().info(f"  Movement 1: {len(self.movement_1_poses)} poses")
        self.get_logger().info(f"  Movement 2: {len(self.movement_2_poses)} poses")
        self.get_logger().info(f"  Movement 3: {len(self.movement_3_poses)} poses")
        self.get_logger().info(f"  Step duration: {self.step_duration}s")
        self.get_logger().info("="*60)
    
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
    
    def result_callback(self, msg):
        self.latest_result = msg.data
        if self.waiting_for_result:
            self.get_logger().info(f" Result: {msg.data}")

    def perform_movement_sequence(self, movement_num, play_audio_cues=False):
        """Perform a movement's pose sequence"""
        movement = self.movements[movement_num]
        poses = movement['poses']
        audios = movement['audios']
        
        audio_interval = len(poses) // len(audios) if play_audio_cues else 0
        audio_idx = 0
        
        for i, pose in enumerate(poses): 
            if play_audio_cues and audio_interval > 0 and i % audio_interval == 0 and audio_idx < len(audios):
                self.play_audio(audios[audio_idx], blocking=False)
                audio_idx += 1
            
            # Move arm
            self.move_arm(pose)
            time.sleep(self.step_duration + 0.1)

    def teach_movement(self, movement_num):
        """Teach a single movement"""
        movement = self.movements[movement_num]
        
        self.get_logger().info("\n" + "─"*60)
        self.get_logger().info(f" Teaching {movement['name']}")
        self.get_logger().info("─"*60)
        
        # Step 1: Silent demonstration
        self.get_logger().info("\n Watch the demonstration (silent)...")
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
        
        self.perform_movement_sequence(movement_num, play_audio_cues=True)

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
            self.play_audio(self.audio_files["good_job"])
        elif self.latest_result == "incorrect":
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
        
        # Introduction
        self.get_logger().info("\n Introduction")
        self.play_audio(self.audio_files["intro"])
        time.sleep(1.0)
        
        # Teach movements 1-3 only
        for movement_num in range(1, 4):
            self.teach_movement(movement_num)
        
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
        
        # Run full session (movements 1-3)
        node.run_full_session()
        
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
