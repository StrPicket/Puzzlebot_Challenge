import os

from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    puzzlebot_description_dir = get_package_share_directory('puzzlebot_description')
    puzzlebot_control_dir = get_package_share_directory('puzzlebot_control')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(puzzlebot_description_dir, 'launch', 'gazebo.launch.py')
        )
    )

    joystick_teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(puzzlebot_control_dir, 'launch', 'joystick_teleop.launch.py')
        )
    )

    simulation_velocity_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(puzzlebot_control_dir, 'launch', 'simulation_velocity_controller.launch.py')
        )
    )

    real_velocity_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(puzzlebot_control_dir, 'launch', 'real_velocity_controller.launch.py')
        )
    )

    return LaunchDescription([
        gazebo_launch,
        joystick_teleop_launch,
        simulation_velocity_controller_launch,
        real_velocity_controller_launch,
    ])

if __name__ == '__main__':
    generate_launch_description()