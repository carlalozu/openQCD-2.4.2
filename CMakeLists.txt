cmake_minimum_required(VERSION 3.18)

project(openQCD
    VERSION 2.4.2
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(OPENQCD_ENABLE_FMA3 "Enable FMA3 (include/avx.h)" OFF)
option(OPENQCD_ENABLE_PM "Enable PM (include/sse.h)" OFF)

message(STATUS "OPENQCD_ENABLE_FMA3 ${OPENQCD_ENABLE_FMA3}")
message(STATUS "OPENQCD_ENABLE_PM ${OPENQCD_ENABLE_PM}")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS C)

add_subdirectory(modules)
add_subdirectory(devel)
