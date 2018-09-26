cmake_minimum_required( VERSION 2.8 )
project( loop_closure )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O0 -g" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
set( DBoW3_INCLUDE_DIRS "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/calculateBoW/DBow3/src" )
set( DBoW3_LIBS "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/calculateBoW/DBow3/lib/libDBoW3.a" )

add_executable( feature_training feature_training.cpp )
target_link_libraries( feature_training ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( loop_closure loop_closure.cpp )
target_link_libraries( loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( gen_vocab gen_vocab_large.cpp )
target_link_libraries( gen_vocab ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( calculate_des_over_threshold calculate_des_over_threshold.cpp )
target_link_libraries( calculate_des_over_threshold ${OpenCV_LIBS} ${DBoW3_LIBS} )