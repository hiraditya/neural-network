cmake_minimum_required(VERSION 2.8)
project(ANN)
include_directories(/home/vmplanet/Documents/AI/ANN)
include_directories(/home/vmplanet/Documents/AI/utilities)
include_directories(/home/vmplanet/Documents/AI)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -pedantic -Wall -DNDEBUG3 -DNDEBUG2 -DNDEBUG1")

add_subdirectory(ANN)
#add_subdirectory(TWEANN)
add_subdirectory(test)
add_subdirectory(test2)
add_subdirectory(utilities)
