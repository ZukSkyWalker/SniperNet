cmake_minimum_required(VERSION 3.10)
project(ptc_player VERSION 1.0)
add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # or another appropriate value


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
# find_package(Boost REQUIRED COMPONENTS filesystem iostreams serialization system)

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
    src/main.cpp
    src/frame.cpp
    src/util.cpp
    libs/cnpy/cnpy.cpp
)

target_link_libraries(
  ptc_player
    ${PCL_LIBRARIES}
)
