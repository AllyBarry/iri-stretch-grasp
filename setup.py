from setuptools import find_packages
from setuptools import setup

package_name = "stretch_floor_perception"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    install_requires=[
        "setuptools",
        "openai",
    ],
    maintainer='Toni Gabas',
    maintainer_email='agabas@iri.upc.edu',
    entry_points={
        'console_scripts': [
            'scan_client = stretch_floor_perception.scan_client:main',
            'stretch_bt = bt_cooking.stretch_bt:main',
        ],
    },

)
