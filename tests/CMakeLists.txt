cmake_minimum_required (VERSION 3.26.3)

find_package(boost REQUIRED)

add_executable(unit_test test_decoders.cpp)
target_compile_features(unit_test PRIVATE cxx_std_17)
target_include_directories(unit_test PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(unit_test cppctcdecoder Eigen3::Eigen kenlm)