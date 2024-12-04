import rclpy
import tf
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from moveit_commander import MoveGroupCommander
from trajectory_msgs.msg import JointTrajectoryPoint


class WorldTrajectoryClient(Node):
    def __init__(
            self,
            node_name: str,
    ):
        super().__init__(node_name=node_name)

        self.move_group = MoveGroupCommander("manipulator")
        self.move_group.set_planner_id("RRTConnectkConfigDefault")
        self.move_group.set_goal_tolerance(0.01)

        self._joint_trajectory_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "joint_trajectory_controller/follow_joint_trajectory",
        )
        while not self._joint_trajectory_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for action server to become available...")
        self.get_logger().info("Action server available.")

    def execute_xyz(self, target_pose: list, sec_from_start: int = 15):
        """
        Set position from world coordinates and keep current orientation.
        """
        current_pose = self.move_group.get_current_pose().pose
        quaternion = [current_pose.orientation.x,
                      current_pose.orientation.y,
                      current_pose.orientation.z,
                      current_pose.orientation.w]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        target_pose.extend(euler)

        self.execute_xyzabc(target_pose, sec_from_start)

    def execute_xyzabc(self, target_pose: list, sec_from_start: int = 15):
        """
        Compute IK for the target pose and send joint positions to the robot.
        """
        self.get_logger().info(f"Received target pose:\n{target_pose}")

        self.move_group.set_pose_target(target_pose)

        plan = self.move_group.plan()
        if not plan.joint_trajectory.points:
            self.get_logger().error("Failed to compute IK solution or trajectory.")
            return

        joint_positions = plan.joint_trajectory.points[0].positions
        self.get_logger().info(f"Computed joint positions: {joint_positions}")

        self.execute_joint_positions(joint_positions, sec_from_start)

    def execute_joint_positions(self, positions: list, sec_from_start: int = 15):
        if len(positions) != 7:
            self.get_logger().error("Invalid number of joint positions.")
            return

        joint_trajectory_goal = FollowJointTrajectory.Goal()
        goal_sec_tolerance = 1
        joint_trajectory_goal.goal_time_tolerance.sec = goal_sec_tolerance

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)
        point.time_from_start.sec = sec_from_start

        for i in range(7):
            joint_trajectory_goal.trajectory.joint_names.append(f"A{i + 1}")

        joint_trajectory_goal.trajectory.points.append(point)

        # send goal
        goal_future = self._joint_trajectory_action_client.send_goal_async(
            joint_trajectory_goal
        )
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected by server.")
            return
        self.get_logger().info("Goal was accepted by server.")

        # wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=sec_from_start + goal_sec_tolerance
        )

        if (
            result_future.result().result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self.get_logger().error("Failed to execute joint trajectory.")
            return

    def move_to_zero_position(self):
        """
        Move the robot to the zero joint position.
        """
        self.execute_xyzabc(
            target_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )


def main(args: list = None) -> None:
    rclpy.init(args=args)

    node = WorldTrajectoryClient("world_trajectory_client")

    target_pose = [-468.0, -93.0, 524.0, 0.0, 0.0, 180.0]

    node.get_logger().info("Moving to target pose.")
    node.execute_xyzabc(target_pose)

    node.get_logger().info("Moving back to zero position.")
    node.move_to_zero_position()

    rclpy.shutdown()
