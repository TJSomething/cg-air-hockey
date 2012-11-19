cmake_minimum_required (VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/deps)

set(LIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(BOX2D_BUILD_EXAMPLES OFF)

set(BOX2D_FOUND true)

set(BOX2D_INCLUDE_DIR ${LIB_ROOT}/Box2D)
ADD_SUBDIRECTORY(${LIB_ROOT}/Box2D)
include_directories(${BOX2D_INCLUDE_DIR})

set(ASSIMP_INCLUDE_DIR ${LIB_ROOT}/assimp/include)
ADD_SUBDIRECTORY(${LIB_ROOT}/assimp)
include_directories(${ASSIMP_INCLUDE_DIR})

set(STB_IMAGE_INCLUDE_DIR ${LIB_ROOT}/stb_image)
ADD_SUBDIRECTORY(${LIB_ROOT}/stb_image)
include_directories(${STB_IMAGE_INCLUDE_DIR})

find_package(PNG REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
FIND_PACKAGE (Boost COMPONENTS system filesystem REQUIRED)

ADD_DEFINITIONS("-std=c++0x")
SET(CMAKE_CXX_FLAGS "")

set(CORELIBS
    ${GLUT_LIBRARY}
    ${OPENGL_LIBRARY}
    ${GLEW_LIBRARY}
    Box2D
    assimp
    ${Boost_LIBRARIES}
    ${PNG_LIBRARY}
    stb_image)

set (HOCKEY_FILES ${HOCKEY_FILES} src/main.cpp src/read_file.cpp)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(hockey ${HOCKEY_FILES})
target_link_libraries(hockey ${CORELIBS})
