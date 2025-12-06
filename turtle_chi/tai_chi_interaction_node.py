#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import os
import subprocess


class TaiChiInteractionNode(Node):
    def __init__(self):
        super().__init__('tai_chi_interaction')
        
        self.declare_parameter('trigger_topic', '/trigger_capture')
        self.declare_parameter('result_topic', '/pose_result')
        self.declare_parameter('arm_topic', '/arm_controller/joint_trajectory')
        self.declare_parameter('audio_dir', '/home/jarch/intro_robo_ws/src/turtle_chi/turtle_chi/')
        self.declare_parameter('use_audio', True)
        
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.arm_topic = self.get_parameter('arm_topic').value
        self.audio_dir = self.get_parameter('audio_dir').value
        self.use_audio = self.get_parameter('use_audio').value
        
        self.get_logger().info("="*60)
        self.get_logger().info("Tai Chi Interaction Node Starting...")
        self.get_logger().info("="*60)
        
        self.trigger_pub = self.create_publisher(Bool, self.trigger_topic, 10)
        self.arm_pub = self.create_publisher(JointTrajectory, self.arm_topic, 10)
        
        self.result_sub = self.create_subscription(
            String,
            self.result_topic,
            self.result_callback,
            10
        )
        
        self.latest_result = None
        self.waiting_for_result = False

        self.final_pose = [0.0, 1.496, -0.407, 0.0] # hands out and knees bent pose

        self.demo_poses = [
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

        self.demo_audios = ["downward_flow.wav", "upward_flow.wav", "back_to_center.wav"]

        self.audio_files = {
            "intro":"intro.wav",
            "downward_flow":"downward_flow.wav",
            "upward_flow":"upward_flow.wav",
            "back_to_center":"back_to_center.wav",
            "good job ur pose worked":"good job ur pose worked.wav",
            "stay_for_evaluate":"stay_for_evaluate.wav",
            "Jacks Mom - very wrong":"Jacks_Mom.wav",
        }

        self.get_logger().info("="*60)
        self.get_logger().info("Interaction Node Ready!")
        self.get_logger().info(f"  Demo has {len(self.demo_poses)} poses for smooth movement")
        self.get_logger().info(f"  Model evaluates final 'hands out' pose only")
        self.get_logger().info(f"  Audio enabled: {self.use_audio}")
        self.get_logger().info("="*60)
    
    def play_audio(self, filename, blocking=True):
        if not self.use_audio:
            self.get_logger().info(f"[Audio: {filename}]")
            return None
        
        audio_path = os.path.join(self.audio_dir, filename)
        
        if not os.path.exists(audio_path):
            self.get_logger().warn(f"Audio file not found: {audio_path}")
            self.get_logger().info(f"[Would play: {filename}]")
            return None

        try:
            self.get_logger().info(f"Playing: {filename}")
            
            if blocking:
                subprocess.run(['aplay', '-q', audio_path], check=True)
                return None
            else:
                process = subprocess.Popen(['aplay', '-q', audio_path])
                return process
                
        except Exception as e:
            self.get_logger().error(f"Error playing audio: {e}")
            return None
    
    def move_arm(self, joint_positions, duration_sec=1.0):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        
        msg.points = [point]
        
        self.arm_pub.publish(msg)
        self.get_logger().info(f"  Moving arm to: {joint_positions}")
    
    def result_callback(self, msg):
        self.latest_result = msg.data
        if self.waiting_for_result:
            self.get_logger().info(f"Received result: {msg.data}")

    def run_teaching_session(self):
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Starting Tai Chi Teaching Session!")
        self.get_logger().info("Target: Hands Out Pose")
        self.get_logger().info("="*60)
        
        self.get_logger().info("\nStep 1: Watch the demonstration")
        self.play_audio(self.audio_files["intro"])
        time.sleep(1.0)
    
        self.get_logger().info("\nStep 2: Robot demonstrating...")
        
        for i, pose in enumerate(self.demo_poses):
            self.get_logger().info(f"  Pose {i+1}/{len(self.demo_poses)}")
            self.move_arm(pose, duration_sec=0.8)
            time.sleep(0.9)

        self.get_logger().info("  Holding final 'hands out' pose...")
        time.sleep(1.5)
        
        self.get_logger().info("\nStep 3: Your turn!")
        time.sleep(1.0)
        
        self.get_logger().info("Follow along slowly and end with hands pushed out...")
        time.sleep(len(self.demo_poses) * 0.9 + 1)
        for i, pose in enumerate(self.demo_poses):
            self.get_logger().info(f"  Pose {i+1}/{len(self.demo_poses)}")
            self.move_arm(pose, duration_sec=0.8)
            audio_num = 0
            if i%2 == 0:
                self.play_audio(self.demo_audios[audio_num])
                audio_num += 1
            time.sleep(0.9)

        self.get_logger().info("\nStep 4: Hold your hands-out pose!")
        time.sleep(0.5)

        self.get_logger().info("Evaluating your pose...")
        self.latest_result = None
        self.waiting_for_result = True

        trigger_msg = Bool()
        trigger_msg.data = True
        self.trigger_pub.publish(trigger_msg)
        
        self.get_logger().info("Waiting for evaluation result...")
        wait_start = time.time()
        while (time.time() - wait_start) < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.waiting_for_result = False
        
        self.get_logger().info("\nStep 5: Feedback")
                    

        self.play_audio(self.audio_files["Jacks Mom - very wrong"])
        # if self.latest_result == "correct":
        #     self.get_logger().info("SUCCESS! Your hands-out pose is correct!")
        #     self.play_audio(self.audio_files["good job ur pose worked"])
        # elif self.latest_result == "incorrect":
        #     self.get_logger().info("Not quite right. Try pushing your arms out more.")
        #     self.play_audio(self.audio_files["Jacks Mom - very wrong"])
        # elif self.latest_result == "no_person":
        #     self.get_logger().warn("No person detected in frame")
        # else:
        #     self.get_logger().warn(f"Unexpected result: {self.latest_result}")
        
        time.sleep(2.0)
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Session Complete!")
        self.get_logger().info("="*60)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TaiChiInteractionNode()
        
        time.sleep(2.0)
        node.run_teaching_session()
        time.sleep(2.0)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
