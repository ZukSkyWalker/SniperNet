cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ptc_player VERSION 1.0)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the Eigen3 cmake directory to CMAKE_MODULE_PATH
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/eigen3/cmake")

set(VTK_DIR "/usr/lib/cmake/vtk-7.1")
set(PCL_INCLUDE_DIRS "/usr/local/include/pcl-1.13")
set(PCL_LIBRARY_DIRS "/usr/local/lib")
find_package(PCL)

# Add the include directories
include_directories(${PCL_INCLUDE_DIRS} inc libs/cnpy)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add the cnpy library
add_library(cnpy STATIC libs/cnpy/cnpy.cpp)

# Add the executable and link the libraries
add_executable(ptc_player src/main.cpp src/frame.cpp)

target_link_libraries(ptc_player cnpy ${PCL_LIBRARIES})

# Optional: enable warnings and warnings-as-errors for better code quality
if(MSVC)
  target_compile_options(ptc_player PRIVATE /W4 /WX)
else()
  target_compile_options(ptc_player PRIVATE -Wall -Wextra -Werror)
endif()

# PCL definitions and compile options
add_definitions(${PCL_DEFINITIONS})
