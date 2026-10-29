"""Full pipeline. Split across machines with the run_* arguments, e.g.
  robot:  ros2 launch speech_task speech_pipeline.launch.py run_compute:=false
  PC:     ros2 launch speech_task speech_pipeline.launch.py run_audio:=false run_tts:=false

Push-to-talk (listen only while SPACE is held in the push-to-talk window):
  ros2 launch speech_task speech_pipeline.launch.py listen_mode:=push_to_talk
  split: robot gets listen_mode:=push_to_talk run_ptt_ui:=false, PC gets listen_mode:=push_to_talk run_audio:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        'run_audio': ('false', 'mic capture + VAD (robot or PC)'),
        'run_compute': ('true', 'transcriber + interpreter (GPU PC)'),
        'run_tts': ('true', 'speak responses'),
        'listen_mode': ('vad', 'vad (always listening) | push_to_talk (only while SPACE is held)'),
        'run_ptt_ui': ('true', 'push_to_talk: open the hold-SPACE window on this machine'),
        'mic_device': ('', 'sounddevice name substring/index, e.g. ReSpeaker'),
        'whisper_model': ('large-v3-turbo', 'faster-whisper model; multilingual. English-only: distil-large-v3 (faster), small.en'),
        'language': ('auto', 'speech language: auto = detect per utterance (en, es, ca), or a code such as en'),
        'translate': ('true', 'translate non-English requests to English before interpreting them'),
        'recipes_dir': ('', 'folder with the recipe .json files (iri-stretch-grasp/config/recipes); empty = find it'),
        'clarify_timeout_s': ('0.0', 'seconds a follow-up answer is joined to an unclear request; 0 = no follow-ups (no spoken replies)'),
        'backend': ('ollama', 'ollama | anthropic | rules'),
        'fallback_backend': ('rules', 'used when the backend fails'),
        'model': ('', 'LLM model name; empty = backend default'),
        'ollama_host': ('http://localhost:11434', ''),
        'run_web': ('true', 'phone/browser page: hold-to-talk with the phone mic, or type (https://<this-pc>:<web_port>)'),
        'web_port': ('8443', 'port of the phone page'),
    }
    decls = [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in args.items()]
    return LaunchDescription(decls + [
        Node(package='speech_task', executable='audio_capture', name='audio_capture', output='screen',
             condition=IfCondition(LC('run_audio')),
             parameters=[{'device': ParameterValue(LC('mic_device'), value_type=str), 'mode': LC('listen_mode')}]),
        Node(package='speech_task', executable='push_to_talk', name='push_to_talk', output='screen',
             condition=IfCondition(PythonExpression(
                 ["'", LC('listen_mode'), "' == 'push_to_talk' and '", LC('run_ptt_ui'), "' == 'true'"]))),
        Node(package='speech_task', executable='transcriber', name='transcriber', output='screen',
             condition=IfCondition(LC('run_compute')),
             parameters=[{'model': LC('whisper_model'), 'language': ParameterValue(LC('language'), value_type=str)}]),
        Node(package='speech_task', executable='interpreter', name='interpreter', output='screen',
             condition=IfCondition(LC('run_compute')),
             parameters=[{'backend': LC('backend'), 'fallback_backend': LC('fallback_backend'),
                          'model': LC('model'), 'ollama_host': LC('ollama_host'), 'translate': LC('translate'),
                          'clarify_timeout_s': ParameterValue(LC('clarify_timeout_s'), value_type=float),
                          'recipes_dir': ParameterValue(LC('recipes_dir'), value_type=str)}]),
        Node(package='speech_task', executable='web_mic', name='web_mic', output='screen',
             condition=IfCondition(LC('run_web')),
             parameters=[{'port': ParameterValue(LC('web_port'), value_type=int)}]),
        Node(package='speech_task', executable='tts', name='tts', output='screen',
             condition=IfCondition(LC('run_tts'))),
    ])
