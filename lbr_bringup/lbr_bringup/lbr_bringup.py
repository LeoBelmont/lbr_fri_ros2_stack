import os
from typing import Dict

import xacro
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


class LBRBringUp:
    model_: str
    robot_name_: str
    sim: bool

    robot_description_: dict
    safety_file_path_: str
    launch_description_: LaunchDescription

    lbr_spinner_node_: Node
    gazebo_spawn_entity_: Node
    controller_manager_: Node
    controller_spawners_: Dict[str, Node]
    robot_state_publisher_: Node
    rviz2_node_: Node

    def __init__(
        self, model: str = "iiwa7", robot_name: str = "lbr", sim: bool = True
    ) -> None:
        self.model_ = model
        self.robot_name_ = robot_name
        self.sim_ = sim

        self.robot_description_ = None
        self.safety_file_path_ = None
        self.launch_description_ = LaunchDescription()

        self.lbr_spinner_node_ = None
        self.gazebo_spawn_entity_ = None
        self.controller_manager_ = None
        self.controller_spawners_ = {}
        self.robot_state_publisher_ = None
        self.rviz2_node_ = None

    @property
    def model(self):
        return self.model_

    @property
    def robot_description(self):
        return self.robot_description_

    @property
    def safety_file_path(self):
        return self.safety_file_path_

    @property
    def launch_description(self):
        return self.launch_description_

    def add_controller_manager(
        self,
        package: str = "lbr_bringup",
        controller_configurations_file: str = "config/lbr_controllers.yml",
    ):
        # Gazebo adds it's own controller manager
        if self.sim_:
            return self
        if not self.robot_description_:
            raise ValueError(
                "Cannot add ros2 control without robot description."
                "Call add_robot_description() first."
            )
        controller_configurations_file_path = os.path.join(
            get_package_share_directory(package), controller_configurations_file
        )
        self.controller_manager_ = Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[self.robot_description_, controller_configurations_file_path],
        )
        self.launch_description_.add_action(self.controller_manager_)
        return self

    def add_controller(self, controller: str = "joint_state_broadcaster"):
        self.controller_spawners_[controller] = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[controller, "--controller-manager", "/controller_manager"],
        )
        controller_spawner_event_handlers = None
        if self.sim_:
            controller_spawner_event_handlers = RegisterEventHandler(
                OnProcessExit(
                    target_action=self.gazebo_spawn_entity_,
                    on_exit=[self.controller_spawners_[controller]],
                )
            )
        else:
            controller_spawner_event_handlers = RegisterEventHandler(
                OnProcessStart(
                    target_action=self.controller_manager_,
                    on_start=[self.controller_spawners_[controller]],
                )
            )
        self.launch_description_.add_action(controller_spawner_event_handlers)
        return self

    def add_robot_state_publisher(self):
        if not self.robot_description_:
            raise ValueError(
                "Cannot add robot state publisher without robot description."  #
                "Call add_robot_description() first."
            )
        joint_state_broadcaster = "joint_state_broadcaster"
        if not joint_state_broadcaster in self.controller_spawners_:
            raise RuntimeError(
                f"The {joint_state_broadcaster} controller needs to be added prior to adding rviz2."
            )
        self.robot_state_publisher_ = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[self.robot_description_],
        )

        # Don't wait on joint_state_broadcaster in case of simulation.
        # Gazebo requires robot_description to load joint_state_broadcaster!
        if self.sim_:
            self.launch_description_.add_action(self.robot_state_publisher_)
        else:
            robot_state_publisher_event_handler = RegisterEventHandler(
                OnProcessExit(
                    target_action=self.controller_spawners_[joint_state_broadcaster],
                    on_exit=[self.robot_state_publisher_],
                )
            )
            self.launch_description.add_action(robot_state_publisher_event_handler)
        return self

    def add_rviz2(
        self,
        package: str = "lbr_description",
        rviz2_config_file: str = "config/config.rviz",
    ):
        if not self.robot_description_:
            raise ValueError(
                "Cannot add rviz2 without robot description. Call add_robot_description() first."
            )
        self.rviz2_node_ = Node(
            package="rviz2",
            executable="rviz2",
            parameters=[self.robot_description_],
            arguments=[
                "-d",
                os.path.join(get_package_share_directory(package), rviz2_config_file),
            ],
        )

        # Don't wait on joint_state_broadcaster in case of simulation.
        # Gazebo requires robot_description to load joint_state_broadcaster!
        if self.sim_:
            rviz2_event_handler = RegisterEventHandler(
                OnProcessExit(
                    target_action=self.gazebo_spawn_entity_, on_exit=[self.rviz2_node_]
                )
            )
        else:
            rviz2_event_handler = RegisterEventHandler(
                OnProcessStart(
                    target_action=self.robot_state_publisher_,
                    on_start=[self.rviz2_node_],
                )
            )
        self.launch_description_.add_action(rviz2_event_handler)
        return self

    def add_robot_description(self):
        self.robot_description_ = {
            "robot_description": xacro.process(
                os.path.join(
                    get_package_share_directory("lbr_description"),
                    f"urdf/{self.model_}/{self.model_}.urdf.xacro",
                ),
                mappings={"robot_name": self.robot_name_, "sim": str(self.sim_)},
            )
        }
        return self

    def add_robot(self):
        if self.sim_:
            return self.add_gazebo()
        else:
            return self.add_lbr_spinner()

    def add_lbr_spinner(
        self,
        package: str = "lbr_fri_ros2",
        safety_file: str = "config/lbr_safety_limits.yml",
        executable: str = "lbr_spinner",
    ):
        self.safety_file_path_ = os.path.join(
            get_package_share_directory(package), safety_file
        )
        self.lbr_spinner_node_ = Node(
            package=package,
            executable=executable,
            parameters=[self.safety_file_path_],
            emulate_tty=True,
            output="screen",
        )
        self.launch_description_.add_action(self.lbr_spinner_node_)
        return self

    def add_gazebo(self):
        gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("gazebo_ros"), "launch/gazebo.launch.py"
                )
            )
        )
        self.gazebo_spawn_entity_ = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=["-topic", "robot_description", "-entity", self.robot_name_],
            output="screen",
        )
        self.launch_description.add_action(gazebo)
        self.launch_description.add_action(self.gazebo_spawn_entity_)
        return self