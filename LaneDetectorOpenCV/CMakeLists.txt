CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(LaneDetector)


SET(CMAKE_CXX_STANDARD 14)
FIND_PACKAGE(OpenCV)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# 设置用debug还是release模式。debug允许断点，而release更快
#set( CMAKE_BUILD_TYPE Debug )
set( CMAKE_BUILD_TYPE Release )

# 设置编译选项
# 允许c++14标准、O3优化、多线程。match选项可避免一些cpu上的问题
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -pthread" )

set(CMAKE_SHARED_LINKER_FLAGS "-L/opt/homebrew/Cellar/qt@5/5.15.2_1/lib")
set(CMAKE_MODULE_LINKER_FLAGS "-L/opt/homebrew/Cellar/qt@5/5.15.2_1/lib")
set(CMAKE_EXE_LINKER_FLAGS  "-L/opt/homebrew/Cellar/qt@5/5.15.2_1/lib")

ADD_EXECUTABLE(LaneDetector main.cpp)
TARGET_LINK_LIBRARIES(LaneDetector ${OpenCV_LIBS})