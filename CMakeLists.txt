cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ptc_player VERSION 1.0)

# Set the C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find PCL package
find_package(PCL 1.8 REQUIRED)

# Add the include directories
include_directories(${PCL_INCLUDE_DIRS} inc libs/cnpy)


# Add the cnpy library
add_library(cnpy STATIC libs/cnpy/cnpy.cpp)

# Add the executable and link the libraries
add_executable(ptc_player src/main.cpp src/visualize.cpp)

target_link_libraries(ptc_player cnpy ${PCL_LIBRARIES})

# Optional: enable warnings and warnings-as-errors for better code quality
if(MSVC)
  target_compile_options(ptc_player PRIVATE /W4 /WX)
else()
  target_compile_options(ptc_player PRIVATE -Wall -Wextra -Werror)
endif()

# PCL definitions and compile options
add_definitions(${PCL_DEFINITIONS})
