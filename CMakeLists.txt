cmake_minimum_required(VERSION 3.5)
project(mn-tp2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# descomentar para debuggear
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -ggdb")

# aguante clang
#set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

add_executable(tp2 src/main.cpp)
add_executable(run_tests test/main.cpp)
add_executable(run_integration_tests test/integration.cpp)

add_custom_target(check
    COMMAND run_tests
    COMMAND run_integration_tests data/train.csv test/test1.in test/test1.expected
    COMMAND run_integration_tests data/train.csv test/test2.in test/test2.expected
)
