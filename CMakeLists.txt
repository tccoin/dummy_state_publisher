cmake_minimum_required(VERSION 3.0.2)
project(dummy_state_publisher)

add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  inekf_msgs
  roscpp
  std_msgs
  tf2
  tf2_geometry_msgs
  cv_bridge
)

# opencv 
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
find_package( OpenCV 4.0 REQUIRED)
# find_package(fbow REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# elas
set(ELAS_LIB_SOURCE
  thirdparty/libelas/descriptor.cpp
  thirdparty/libelas/elas.cpp
  thirdparty/libelas/filter.cpp
  thirdparty/libelas/matrix.cpp
  thirdparty/libelas/triangle.cpp
  )
add_library(elas SHARED ${ELAS_LIB_SOURCE})
include_directories("thirdparty")
## Declare a catkin package
catkin_package()

## Build execs
list(APPEND thirdparty_libraries
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  ${fbow_LIBS}
)
file(GLOB TARGET_SRC
  thirdparty/libelas/*.cpp
  src/depth_generator.cpp
  src/state_publisher_node.cpp
)
include_directories(include ${catkin_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${TARGET_SRC})
target_link_libraries(${PROJECT_NAME} ${thirdparty_libraries})
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})

# add_executable(fbowGen src/fbowGen/fbow_generator.cpp)
# target_link_libraries(fbowGen ${thirdparty_libraries})
# add_dependencies(fbowGen ${catkin_EXPORTED_TARGETS})