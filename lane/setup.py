from setuptools import find_packages, setup

package_name = 'lane'

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
    maintainer='ljh',
    maintainer_email='ljh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane1 = lane.lane1:main',
            'lane2 = lane.lane2:main',
            'lane3 = lane.lane3:main',
            'lane4 = lane.lane4:main',
            'lane1white = lane.lane1white:main',
            'lane3white = lane.lane3white:main',
            'lanehsv = lane.lanehsv:main',
            'lane5 = lane.lane5:main',
            'lane6 = lane.lane6:main',
            'lane7 = lane.lane7:main',
        ],
    },
)

