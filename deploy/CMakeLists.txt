cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libyesno)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wno-narrowing")

add_library(yesno SHARED yesno.cc)
target_link_libraries(yesno "${TORCH_LIBRARIES}")
set_property(TARGET yesno PROPERTY CXX_STANDARD 14)

add_executable(example-app example-app.cc)
target_link_libraries(example-app yesno)
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
