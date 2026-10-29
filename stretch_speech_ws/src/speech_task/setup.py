from glob import glob

from setuptools import find_packages, setup

package_name = 'speech_task'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='userlab',
    maintainer_email='1056862@students.wits.ac.za',
    description='Speech to structured high-level robot tasks (clarify / infeasible aware).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_capture = speech_task.audio_capture_node:main',
            'transcriber = speech_task.transcriber_node:main',
            'interpreter = speech_task.interpreter_node:main',
            'text_input = speech_task.text_input_node:main',
            'tts = speech_task.tts_node:main',
            'wav_publisher = speech_task.wav_publisher_node:main',
            'push_to_talk = speech_task.push_to_talk_node:main',
            'web_mic = speech_task.web_mic_node:main',
        ],
    },
)
