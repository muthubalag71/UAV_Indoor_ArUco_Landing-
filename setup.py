
from setuptools import find_packages, setup

package_name = 'aruco_landing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	'camera_pub = aruco_landing.camera_pub:main',
	'x500mavros = aruco_landing.x500mavros:main',
	'uav_box_ground_ref = aruco_landing.uav_box_ground_ref:main',
	'camera_pub_720 = aruco_landing.camera_pub_720:main',
	'aruco_web_pose = aruco_landing.aruco_web_pose:main',
	'web_stream_light = aruco_landing.web_stream_light:main',
	'demo_detection_and_movement = aruco_landing.demo_detection_and_movement:main',
	'Drone_presion_test = aruco_landing.Drone_presion_test:main',
	'improved_uav_box = aruco_landing.improved_uav_box:main',
	'uav_box_viewer = aruco_landing.uav_box_viewer:main',
	'aruco_headless_guidance = aruco_landing.aruco_headless_guidance:main',
	'aruco_mission_control = aruco_landing.aruco_mission_control:main',
	'uav_main_supervisor = aruco_landing.uav_main_supervisor:main',
        'box_search_node = aruco_landing.box_search_node:main',
	'aruco_axis_test_node = aruco_landing.aruco_axis_test_node:main',
	'uav_axis_calibration = aruco_landing.uav_axis_calibration:main',
	'aruco_interupt = aruco_landing.aruco_interupt:main',
	'aruco_landing_Test1 = aruco_landing.aruco_landing_Test1:main',
	'aruco_axis_calibration = aruco_landing.aruco_axis_calibration:main',
	'axis_pulse_coupling = aruco_landing.axis_pulse_coupling:main',
	'aruco_landing_Test2 = aruco_landing.aruco_landing_Test2:main',
	'body_axis_check_2 = aruco_landing.body_axis_check_2:main',
        ],
    },
)
