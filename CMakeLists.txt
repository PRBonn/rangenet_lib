cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(rangenet_lib)

# handle debug vs release mode
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

## Use C++11
set (CMAKE_CXX_STANDARD 11)

# check flags
message("Build type: ${CMAKE_BUILD_TYPE}")

## Find catkin macros and libraries
find_package(catkin REQUIRED)

# libyaml-cpp (required)
find_package (yaml-cpp)
if (yaml-cpp_FOUND)
  message("YAML Libs: ${YAML_CPP_LIBRARIES}")
  message("YAML Headers: ${YAML_CPP_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "yaml-cpp NOT FOUND\n")
endif (yaml-cpp_FOUND)

# get boost (required)
find_package(Boost COMPONENTS program_options filesystem)
if(Boost_FOUND)
  message("Boost Libs: ${Boost_LIBRARIES}")
  message("Boost Headers: ${Boost_INCLUDE_DIRS}\n")
else()
  message(FATAL_ERROR "Boost NOT FOUND\n")
endif()


####################################
## TensorRT specific configuration ##
####################################
# this package libs and includes for TensorRT
option(TENSORRT_FOUND "TensorRT installed on system?" OFF)

# first try to find cuda and tensorrt-related stuff (nvinfer, onnx, etc)
find_package(TensorRT PATHS cmake)
if (TENSORRT_FOUND)
  message("TensorRT SUCCESS!\n")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT FOUND\n")
  set(TENSORRT_FOUND OFF)
endif (TENSORRT_FOUND)

# complain if no backend is installed
if(NOT TENSORRT_FOUND)
  message(FATAL_ERROR
  "TensorRT must be installed
  TENSORRT_FOUND ${TENSORRT_FOUND}\n")
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    rangenet_external
    rangenet_lib
  CATKIN_DEPENDS
  DEPENDS
    YAML_CPP
  CFG_EXTRAS
    tensorrt-config.cmake
)

#########################
## Install cmake finds ##
#########################
install(DIRECTORY cmake
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

###########################
## Install header config ##
###########################
include_directories(
  example
  include
  ${rangenet_external_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${TENSORRT_INCLUDE_DIRS}
)

# turn config file into flags (sets TENSORRT_FOUND and TORCH_FOUND)
configure_file(include/external.hpp.in ${PROJECT_SOURCE_DIR}/include/external.hpp @ONLY)

# Mark cpp header files for installation
install(
  DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

# make wrapper lib
add_library(rangenet_external src/external.cpp)

## Declare a cpp library
# define the network library (compiling tf and trt conditionally)
set(rangenet_lib_SOURCES src/net.cpp src/selector.cpp)

if(TENSORRT_FOUND)
  message("Building TensorRT")
  list(APPEND rangenet_lib_SOURCES src/netTensorRT.cpp)
endif()

# make the library and link stuff to it
add_library(rangenet_lib ${rangenet_lib_SOURCES})
target_link_libraries(rangenet_lib
    rangenet_external
    ${catkin_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${TENSORRT_LIBRARIES})

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS rangenet_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

###################
## build example ##
###################

option(EXAMPLE "Building example" ON)

if(EXAMPLE)

  # get opencv (required)
  find_package(OpenCV COMPONENTS core viz)
  if(OpenCV_FOUND)
    message("Building example...")
    message("OpenCV Libs: ${OpenCV_LIBS}")
    set(OpenCV_LIBRARIES ${OpenCV_LIBS})
    message("OpenCV Headers: ${OpenCV_INCLUDE_DIRS}\n")

    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(infer example/infer.cpp)
    target_link_libraries(infer   rangenet_lib
                                  ${catkin_LIBRARIES}
                                  ${OpenCV_LIBRARIES}
                                  ${Boost_LIBRARIES}
                                  ${YAML_CPP_LIBRARIES}
                                  ${TENSORRT_LIBRARIES}
                                  )
    install(
      TARGETS
      infer
      DESTINATION bin/
    )
  endif()
else()
  message("Not building example.")
endif()




