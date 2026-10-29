cmake_minimum_required(VERSION 3.8)
project(speech_task_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioUtterance.msg"
  "msg/Transcript.msg"
  "msg/TaskCommand.msg"
  "srv/ParseText.srv"
  "srv/ParseToJson.srv"
  DEPENDENCIES std_msgs diagnostic_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
