cmake_minimum_required(VERSION 2.8)

include_directories(/home/vmplanet/Documents/AI/TWEANN)
include_directories(/home/vmplanet/Documents/AI)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -pedantic -Wall -DNDEBUG3 -DNDEBUG2 -DNDEBUG1")

set(basic_verification
  basic_verification.cpp
)

set(dfs
  dfs.cpp
)

set(boolean
  boolean.cpp
)

#add_executable(basic_verification ${basic_verification})
#add_executable(dfs ${dfs})
add_executable(boolean ${boolean})

