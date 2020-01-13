cmake_minimum_required(VERSION 3.4.1)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})
#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -flax-vector-conversions")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions -flax-vector-conversions")
#set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -Wl,--gc-sections")

include_directories(src/main/cpp/include
                    src/main/cpp/)

add_library(lib_opencv STATIC IMPORTED ) #表示创建一个导入库，静态方式 引入libopencv_java3.so文件
set_target_properties(lib_opencv
                       PROPERTIES
                       IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so
                       )
					   
# include_directories(${CMAKE_SOURCE_DIR}/inc_opencv)
# add_definitions(-std=c++11)
# set(OpenCV_STATIC ON)
# set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni)
# find_package(OpenCV 3 REQUIRED)			
	 							  
add_library(MNN SHARED IMPORTED)
add_library(MNN_CL SHARED IMPORTED)
#add_library(MNN_GL SHARED IMPORTED)
set_target_properties(MNN
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN.so)
set_target_properties(MNN_CL
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN_CL.so)
#set_target_properties(MNN_GL
#  PROPERTIES IMPORTED_LOCATION
#  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN_GL.so)

		
		
  
add_library(facedetect SHARED src/main/cpp/Ultra_jni.cpp
                              src/main/cpp/UltraFace.cpp
                              src/main/cpp/net.cpp)


find_library(log-lib log)
find_library( jnigraphics-lib jnigraphics )
add_definitions(-DMNN_USE_LOGCAT)
target_link_libraries(facedetect
                       MNN
                       MNN_CL
		       MNN_GL
		       lib_opencv
                       jnigraphics
                       z
                       ${log-lib})


