cmake_minimum_required(VERSION 2.8)
project(ANN)

include_directories(ANN)
include_directories(utilities)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -pedantic -Wall -DNDEBUG3 -DNDEBUG2 -DNDEBUG1")

add_subdirectory(ANN)
#add_subdirectory(TWEANN)
add_subdirectory(testANN)
#add_subdirectory(testTWEANN)
add_subdirectory(utilities)
