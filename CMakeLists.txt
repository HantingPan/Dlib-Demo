cmake_minimum_required(VERSION 3.10)
project(dlib_detect)

#opencv
list(APPEND CMAKE_PREFIX_PATH "./thirdParty/opencv/")
find_package(OpenCV REQUIRED)

#dlib
list(APPEND CMAKE_PREFIX_PATH "./thirdParty/dlib/lib/cmake/dlib")
find_package(dlib REQUIRED)

add_executable(dlib_detect main.cpp)
target_link_libraries(dlib_detect ${OpenCV_LIBS} dlib::dlib)


