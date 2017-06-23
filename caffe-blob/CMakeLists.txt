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

## Protobuf
find_package(Protobuf REQUIRED)
message(${PROTOBUF_LIBRARY})

## OpenCL
find_package(OpenCL REQUIRED)


## blob-reader
add_executable(blob-reader blob-reader.cpp)
target_include_directories(blob-reader PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(blob-reader ${Boost_LIBRARIES})
target_include_directories(blob-reader PUBLIC ${Caffe_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIR})
target_link_libraries(blob-reader ${Caffe_LIBS} ${PROTOBUF_LIBRARY})

## max-pool
add_executable(max-pool max-pool.cpp)
target_include_directories(max-pool PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(max-pool ${Boost_LIBRARIES})
target_include_directories(max-pool PUBLIC ${Caffe_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIR})
target_link_libraries(max-pool ${Caffe_LIBS} ${PROTOBUF_LIBRARY})

## conv2-pool2
add_custom_target(conv2-pool2
    COMMAND [ -f ./conv2.bin ] || cp -f ../../test-data/bvlc-caffenet/conv2.bin ./
    COMMAND ./max-pool)

# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
