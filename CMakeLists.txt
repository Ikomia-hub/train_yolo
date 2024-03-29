cmake_minimum_required(VERSION 3.16)
set(PLUGIN_VERSION 1.4.0)
project(train_yolo VERSION ${PLUGIN_VERSION} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(UNIX)
    set(CMAKE_INSTALL_PREFIX $ENV{HOME}/Ikomia/Plugins/C++)
endif()

if(WIN32)
    set(CMAKE_INSTALL_PREFIX $ENV{USERPROFILE}/Ikomia/Plugins/C++)
endif()

set(CMAKE_INSTALL_PLUGIN_DIR ${CMAKE_INSTALL_PREFIX}/train_yolo)
set(IKOMIA_CORE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../IkomiaCore)

# Set up AUTOMOC and some sensible defaults for runtime execution
# When using Qt 6.3, you can replace the code block below with
# qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)
include(GNUInstallDirs)
include(${IKOMIA_CORE_DIR}/Versions.cmake)
include(${IKOMIA_CORE_DIR}/LocalSettings.cmake)

# Qt
find_package(Qt5 REQUIRED COMPONENTS Core Gui Sql Widgets)

# Python
if(CENTOS7)
    set(Python3_DIR "/work/shared/local")
endif()

find_package(Python3 ${PYTHON_VERSION_DOT} EXACT REQUIRED COMPONENTS Interpreter Development NumPy)

# OpenMP
find_package(OpenMP)

# OpenCV
if(WIN32)
    if(${IK_CPU})
        set(OpenCV_DIR "C:/Program Files/OpenCV/cpu/x64/vc16/lib")
    else()
        set(OpenCV_DIR "C:/Program Files/OpenCV/cuda/x64/vc16/lib")
    endif()
endif()

find_package(OpenCV ${OPENCV_VERSION} REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

if(CENTOS7)
    set(BUILD_SHARED_LIBS ON)
    set(BOOST_PYTHON_TARGET Boost::python)
    find_package(Boost REQUIRED COMPONENTS system filesystem python)
else()
    set(BOOST_PYTHON_TARGET Boost::python${PYTHON_VERSION_NO_DOT})
    find_package(Boost REQUIRED COMPONENTS system filesystem python${PYTHON_VERSION_NO_DOT})
endif()

add_library(train_yolo SHARED
    YoloTrain.hpp
    YoloTrainGlobal.hpp
    YoloTrainProcess.cpp
    YoloTrainProcess.h
    YoloTrainWidget.cpp
    YoloTrainWidget.h
)

set_target_properties(train_yolo PROPERTIES
    VERSION ${PLUGIN_VERSION}
    SOVERSION ${PLUGIN_VERSION}
)

target_compile_definitions(train_yolo PRIVATE
    BOOST_ALL_NO_LIB
    QT_DEPRECATED_WARNINGS
    YOLOTRAIN_LIBRARY
)

target_compile_features(train_yolo PRIVATE cxx_std_14)

if(MSVC)
    add_compile_options(
        /arch:AVX2
        -D_CRT_SECURE_NO_WARNINGS
    )
endif()

target_include_directories(train_yolo PRIVATE
    # Python
    ${Python3_INCLUDE_DIRS}
    # Numpy
    ${Python3_NumPy_INCLUDE_DIRS}
    # Ikomia
    ${IKOMIA_CORE_DIR}/Build/include/Core
    ${IKOMIA_CORE_DIR}/Build/include/DataProcess
    ${IKOMIA_CORE_DIR}/Build/include/Utils
)

if(WIN32)
    include_directories(
        # Boost
        ${Boost_INCLUDE_DIRS}/Boost/include/boost-${BOOST_VERSION}
        # OpenCL
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include"
        #OpenCV
        ${OpenCV_INCLUDE_DIRS}
    )
endif()

target_link_directories(train_yolo PRIVATE
    ${IKOMIA_CORE_DIR}/Build/lib
)

target_link_libraries(train_yolo PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    OpenMP::OpenMP_CXX
    Python3::Python
    Boost::filesystem
    ${BOOST_PYTHON_TARGET}
    opencv_core
    opencv_dnn
    opencv_imgproc
    ikUtils
    ikCore
    ikDataProcess
)

install(TARGETS train_yolo
    LIBRARY DESTINATION ${CMAKE_INSTALL_PLUGIN_DIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_PLUGIN_DIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PLUGIN_DIR}
)

# Install darknet executable
if(WIN32)
    set(DARKNET_FILES
        ${CMAKE_CURRENT_LIST_DIR}/../../../darknet/darknet.exe
        ${CMAKE_CURRENT_LIST_DIR}/../../../darknet/3rdparty/pthreads/bin/pthreadGC2.dll
        ${CMAKE_CURRENT_LIST_DIR}/../../../darknet/3rdparty/pthreads/bin/pthreadVC2.dll
    )
endif()

if(UNIX)
    set(DARKNET_FILES
        ${CMAKE_CURRENT_LIST_DIR}/../../../darknet/buildDarknet/darknet
    )
endif()

install(FILES ${DARKNET_FILES}
    DESTINATION ${CMAKE_INSTALL_PLUGIN_DIR}
)
