cmake_minimum_required (VERSION 3.26.3)

find_package(boost REQUIRED)
add_library(cppctcdecoder decoder.cpp alphabet.cpp language_model.cpp)
target_compile_features(cppctcdecoder PRIVATE cxx_std_17)
target_include_directories(cppctcdecoder PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/externals/kenlm)
target_link_libraries (cppctcdecoder Eigen3::Eigen kenlm Boost::boost)

target_compile_features(cppctcdecoder PRIVATE cxx_std_17)