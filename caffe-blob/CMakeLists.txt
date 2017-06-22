cmake_minimum_required(VERSION 3.0)
project(caffe-blob)
set(CMAKE_PROJECT_NAME caffe-blob)
set(CMAKE_BUILD_TYPE DEBUG)

## for default arch ##
set(ARCH x86_64)

# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SWNR_SRCS)
add_definitions("-std=c++11")

## boost
# set(Boost_DEBUG 1)
find_package(Boost COMPONENTS program_options REQUIRED)
# message(${Boost_INCLUDE_DIRS})

## Caffe
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Caffe REQUIRED)
# message(${Caffe_FOUND})

## blob-reader
add_executable(blob-reader blob-reader.cpp)
target_include_directories(blob-reader PUBLIC ${Boost_INCLUDE_DIRS} ${Caffe_INCLUDE_DIR})
target_link_libraries(blob-reader ${Boost_LIBRARIES} ${Caffe_LIBS})

# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
