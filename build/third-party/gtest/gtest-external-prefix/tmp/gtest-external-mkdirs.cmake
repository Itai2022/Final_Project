# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/src/gtest-external"
  "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/src/gtest-external-build"
  "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix"
  "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/tmp"
  "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/src/gtest-external-stamp"
  "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/src"
  "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/src/gtest-external-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/src/gtest-external-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/winter/Desktop/pmsc-exercise-6-whiletrue/build/third-party/gtest/gtest-external-prefix/src/gtest-external-stamp${cfgdir}") # cfgdir has leading slash
endif()
