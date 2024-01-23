cmake_minimum_required (VERSION 3.26.3)
project (pyctcdecode)
# set(CMAKE_VERBOSE_MAKEFILE ON)
# TODO clean up appending to cmake prefix path 
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")
find_package (Eigen3 3.4 REQUIRED NO_MODULE)

# kenlm
add_subdirectory(externals/kenlm)

# boost
# include(${CMAKE_CURRENT_SOURCE_DIR}/externals/boost.cmake)
find_package(boost REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(src)
add_subdirectory(tests)

add_executable(main main.cpp)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src externals/kenlm)
target_link_libraries(main cppctcdecoder)

# add_executable(main main.cpp src/decoder.cpp src/alphabet.cpp src/language_model.cpp)
# target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src externals/kenlm)
# target_link_libraries (main Eigen3::Eigen kenlm Boost::boost)