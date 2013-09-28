cmake_minimum_required(VERSION 2.8)

include_directories(/home/vmplanet/Documents/AI/ANN)
include_directories(/home/vmplanet/Documents/AI/utilities)
include_directories(/home/vmplanet/Documents/AI)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -pedantic -Wall -DNDEBUG3 -DNDEBUG2 -DNDEBUG1")

set(setup setup.cpp)
set(FFNetSetup FFNetSetup.cpp)
set(Boolean Boolean.cpp)
set(Boolean2 Boolean2.cpp)
set(Boolean2 Boolean2.cpp)
set(RandomInputs RandomInputs.cpp)

#add_executable(setup ${setup})
#add_executable(FFNetSetup ${FFNetSetup})
#add_executable(Boolean ${Boolean})
#add_executable(Boolean2 ${Boolean2})
add_executable(RandomInputs ${RandomInputs})

