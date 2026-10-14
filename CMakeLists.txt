cmake_minimum_required(VERSION 3.8)
project(stretch_floor_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject3D.msg"
  "msg/DetectedObject3DArray.msg"
  "srv/DetectFloorObjects3D.srv"
  "srv/GraspPoint.srv"
  SKIP_INSTALL
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs std_msgs vision_msgs
)

ament_python_install_package(${PROJECT_NAME})

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py/${PROJECT_NAME}
  DESTINATION ${PYTHON_INSTALL_DIR}
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}__rosidl_generator_c.so
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}__rosidl_typesupport_c.so
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}__rosidl_typesupport_cpp.so
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}__rosidl_typesupport_fastrtps_c.so
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}__rosidl_typesupport_fastrtps_cpp.so
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}__rosidl_typesupport_introspection_c.so
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}__rosidl_typesupport_introspection_cpp.so
  DESTINATION lib
)

install(PROGRAMS
  stretch_floor_perception/floor_object_detector.py
  stretch_floor_perception/scan_client.py
  stretch_floor_perception/grasp_point.py
  stretch_floor_perception/grasp_carry.py
  stretch_floor_perception/topdown_grasp_point.py
  stretch_floor_perception/handover.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
