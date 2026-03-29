cmake_minimum_required(VERSION 3.16)
project(neat_detector CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(pybind11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include(FetchContent)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(json)

# rt-cqt: header-only SlidingCqt — only fetch source, don't build their targets
FetchContent_Declare(
    rtcqt
    GIT_REPOSITORY https://github.com/jmerkt/rt-cqt.git
    GIT_TAG        HEAD
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(rtcqt)
if(NOT rtcqt_POPULATED)
    FetchContent_Populate(rtcqt)
endif()

file(GLOB_RECURSE SOURCES src/*.cpp)
# Exclude bindings.cpp from main executable
list(FILTER SOURCES EXCLUDE REGEX ".*bindings\\.cpp$")

add_executable(neat_detector ${SOURCES})

target_include_directories(neat_detector PRIVATE
    src
    ${SNDFILE_INCLUDE_DIRS}
    ${rtcqt_SOURCE_DIR}/include
)
target_link_libraries(neat_detector PRIVATE
    ${SNDFILE_LIBRARIES}
    nlohmann_json::nlohmann_json
)
target_compile_options(neat_detector PRIVATE -O3 -ffast-math -flto -march=native -Wall -Wextra)

pybind11_add_module(neat_fitness
    src/bindings.cpp
    src/fitness.cpp
    src/neat/genome.cpp
    src/neat/network.cpp
    src/neat/population.cpp
    src/audio/dataset.cpp
    src/audio/processing.cpp
)

target_include_directories(neat_fitness PRIVATE
    src
    ${SNDFILE_INCLUDE_DIRS}
    ${rtcqt_SOURCE_DIR}/include
)
target_link_libraries(neat_fitness PRIVATE
    ${SNDFILE_LIBRARIES}
    nlohmann_json::nlohmann_json
)
target_compile_options(neat_fitness PRIVATE -O3 -ffast-math -march=native)

option(BUILD_BENCH "Build micro-benchmark" OFF)
if(BUILD_BENCH)
    add_executable(neat_bench
        bench/bench.cpp
        src/neat/network.cpp
        src/neat/genome.cpp
        src/neat/population.cpp
    )
    target_include_directories(neat_bench PRIVATE
        src
        ${rtcqt_SOURCE_DIR}/include
    )
    target_compile_options(neat_bench PRIVATE -O3 -march=native)
endif()
