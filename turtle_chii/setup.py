from setuptools import find_packages, setup

package_name = 'turtle_chi'

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
    maintainer='poojavegesna',
    maintainer_email='poojavegesna@uchicago.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'camera-images = turtle_chi.camera_images:main',
		'pose-node = turtle_chi.tai_pose_node:main',
		'interaction-node = turtle_chi.interaction_node:main',
        ],
    },
)
