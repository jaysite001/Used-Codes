# Install script for directory: /media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "main")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/lib/libDBoW3.a")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "main")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/DBoW3" TYPE FILE FILES
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/BowVector.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/Database.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/DBoW3.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/DescManip.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/exports.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/FClass.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/FeatureVector.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/FORB.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/FSGD.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/QueryResults.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/quicklz.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/ScoringObject.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/timers.h"
    "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/Vocabulary.h"
    )
endif()

