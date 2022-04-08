#-------------------------------------------------
#
# Project created by QtCreator 2019-03-20T09:26:01
#
#-------------------------------------------------
QT += core gui widgets sql
TARGET = train_yolo

win32: DESTDIR = $$(USERPROFILE)/Ikomia/Plugins/C++/$$TARGET
unix: DESTDIR = $$(HOME)/Ikomia/Plugins/C++/$$TARGET

TEMPLATE = lib
CONFIG += plugin

DEFINES += YOLOTRAIN_LIBRARY BOOST_ALL_NO_LIB

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

include(../../../IkomiaCore/IkomiaPluginsCpp.pri)

HEADERS += \
    YoloTrain.hpp \
    YoloTrainGlobal.hpp \
    YoloTrainProcess.h \
    YoloTrainWidget.h

SOURCES += \
    YoloTrainProcess.cpp \
    YoloTrainWidget.cpp

# OpenCV
win32:CONFIG(release, debug|release): LIBS += -lopencv_core$${OPENCV_VERSION} -lopencv_imgproc$${OPENCV_VERSION} -lopencv_dnn$${OPENCV_VERSION}
else:win32:CONFIG(debug, debug|release): LIBS += -lopencv_core$${OPENCV_VERSION}d -lopencv_imgproc$${OPENCV_VERSION}d -lopencv_dnn$${OPENCV_VERSION}d
unix:!macx: LIBS += -lopencv_core -lopencv_imgproc -lopencv_dnn
macx: LIBS += -lopencv_core.$${OPENCV_VERSION} -lopencv_imgproc.$${OPENCV_VERSION} -lopencv_dnn.$${OPENCV_VERSION}

# Ikomia libs
LIBS += $$link_utils()
LIBS += $$link_core()
LIBS += $$link_dataprocess()

# DEPLOYMENT
unix:!macx {
bundleDarknet.path = $$DESTDIR
bundleDarknet.commands += $$quote(cp -rf $$PWD/../../../darknet/buildDarknet/darknet $$DESTDIR $$escape_expand(\n\t))
#bundleDarknet.commands += $$quote(patchelf --set-rpath \'\$$ORIGIN:\$$ORIGIN/../../../App/lib\' $$DESTDIR/darknet $$escape_expand(\n\t))
INSTALLS += bundleDarknet
}

macx {
INSTALLS += makeDeploy
}

DISTFILES += \
    Icon/icon.png
