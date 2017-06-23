cmake_minimum_required(VERSION 3.0)
project(empty)
set(CMAKE_PROJECT_NAME empty)
set(CMAKE_BUILD_TYPE DEBUG)

## for default arch ##
set(ARCH x86_64)

# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SWNR_SRCS)
add_definitions("-std=c++11")

## boost
# set(Boost_DEBUG 1)
find_package(Boost COMPONENTS program_options REQUIRED)

## OpenCL
find_package(OpenCL REQUIRED)

## empty
add_executable(empty empty.cpp)
target_link_libraries(empty OpenCL)
target_link_libraries(empty ${Boost_LIBRARIES})

## addone
add_executable(addone addone.cpp)
target_link_libraries(addone OpenCL)
target_link_libraries(addone ${Boost_LIBRARIES})

# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
