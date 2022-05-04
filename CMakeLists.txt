cmake_minimum_required(VERSION 3.0.2)
project(dummy_state_publisher)

add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  inekf_msgs
  roscpp
  std_msgs
)

## Declare a catkin package
catkin_package()

## Build execs
include_directories(include ${catkin_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} src/state_publisher_node.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})