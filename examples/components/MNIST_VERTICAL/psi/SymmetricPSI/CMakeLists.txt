cmake_minimum_required(VERSION 3.13)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")

project(SymmetricPSI VERSION 1.0.0 LANGUAGES CXX C)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(APSI CONFIG REQUIRED)

pybind11_add_module(SymmetricPSI STATIC psi.cpp)
target_link_libraries(SymmetricPSI PRIVATE APSI::apsi Python3::Python pybind11::lto pybind11::embed pybind11::module pybind11::headers)
set_target_properties(SymmetricPSI PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME "SymmetricPSI")
