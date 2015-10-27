cmake_minimum_required(VERSION 2.8)
project(neuralnet)

set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(neuralnet neuralnet.cpp Feature.cpp Instance.cpp Dataset.cpp LogisticRegression.cpp)