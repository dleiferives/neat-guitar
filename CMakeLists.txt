cmake_minimum_required(VERSION 3.16)
project(neat_detector CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(json)

file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(neat_detector ${SOURCES})

target_include_directories(neat_detector PRIVATE
    src
    ${FFTW3F_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)
target_link_libraries(neat_detector PRIVATE
    ${FFTW3F_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    nlohmann_json::nlohmann_json
)
target_compile_options(neat_detector PRIVATE -O3 -march=native -Wall -Wextra)
