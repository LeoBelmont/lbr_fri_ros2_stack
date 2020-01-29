#!/usr/bin/env python3
import os
import sys
import argparse

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import ThisLaunchFileDir
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def get_args():
    parser = argparse.ArgumentParser(description='Argument parser for ROS2 package lbr_gazebo.')
    parser.add_argument('-m', '--model', type=str, default='med7', help='Available models are iiwa7, iiwa14, med7, and med 14.')
    
    return parser.parse_args(sys.argv[4:])

def generate_launch_description():
    # read arguments from shell, e.g. the model
    args = get_args()
    model = args.model

    # export the models directory as environment variable for Gazebo
    models_dir = os.path.join(get_package_share_directory('lbr_gazebo'), 'models')

    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] = os.environ['GAZEBO_MODEL_PATH'] + ':' + models_dir
    else:
        os.environ['GAZEBO_MODEL_PATH'] = models_dir

    # launch Gazebo with model and the robot_state_publisher to turn joint positions to transforms
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    world_file_name = model + '.world'
    world = os.path.join(get_package_share_directory('lbr_gazebo'), 'worlds', world_file_name)

    urdf_file_name = model + '.urdf'
    urdf = os.path.join(get_package_share_directory('lbr_description'), 'urdf', urdf_file_name)

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_init.so'],
            output='screen'),

        ExecuteProcess(
            cmd=['ros2', 'param', 'set', '/gazebo', 'use_sim_time', use_sim_time],
            output='screen'),

        Node(
            package='robot_state_publisher',
            node_executable='robot_state_publisher',
            arguments=[urdf])
    ])
