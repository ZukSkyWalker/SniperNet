cmake_minimum_required(VERSION 3.10)
project(ptc_player VERSION 1.0)
add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # or another appropriate value

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for the GUI
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Compiler optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/libs/cnpy
    ${CMAKE_SOURCE_DIR}/libs/nlohmann
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(
  ptc_player
    src/frame_tester.cpp
    src/frame.cpp
    src/visualize.cpp
    src/util.cpp
    libs/cnpy/cnpy.cpp
)

target_link_libraries(
  ptc_player
    ${PCL_LIBRARIES}
)
