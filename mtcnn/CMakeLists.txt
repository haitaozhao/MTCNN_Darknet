CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(mtcnn)

## darknet
SET(DARKNET ${PROJECT_SOURCE_DIR}/../darknet)
INCLUDE_DIRECTORIES(
	${DARKNET}/include
	${DARKNET}/src
)
LINK_DIRECTORIES(${DARKNET})

## opencv
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

## source code
INCLUDE_DIRECTORIES(./include)
AUX_SOURCE_DIRECTORY(./src SRC_FILES)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_FILES})
TARGET_LINK_LIBRARIES(
	${PROJECT_NAME}
	${OpenCV_LIBS}
	darknet
	m
	pthread
)
