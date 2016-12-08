cmake_minimum_required (VERSION 2.8)

project (wfh)

add_subdirectory(libjpeg)

set(CMAKE_BUILD_TYPE Debug)

add_definitions("-DUSE_WEBSOCKET")

set (SOURCE_FILES
    mongoose/mongoose.c
    cpujpeg.cpp
    wfh_main.cpp
    wfh_panel.cpp
    )

include_directories(
    libjpeg
    ${CMAKE_CURRENT_BINARY_DIR}/libjpeg
    )



if (UNIX)
    set(LIBS
        dl
        pthread
        )
endif (UNIX)

link_directories(
    )
add_executable(wfh ${SOURCE_FILES})
target_link_libraries(wfh
    libjpeg
    ${LIBS}
    )
