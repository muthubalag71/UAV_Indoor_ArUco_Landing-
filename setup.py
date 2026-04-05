
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
	'x500mavros = aruco_landing.x500mavros:main',
	'camera_pub_720 = aruco_landing.camera_pub_720:main',
	'web_stream_light = aruco_landing.web_stream_light:main',
	'uav_box_viewer = aruco_landing.uav_box_viewer:main',
	'aruco_headless_guidance = aruco_landing.aruco_headless_guidance:main',
	'aruco_landing_test4 = aruco_landing.aruco_landing_Test4:main',
        ],
    },
)
