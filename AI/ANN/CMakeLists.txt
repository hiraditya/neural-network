cmake_minimum_required(VERSION 2.8)

set(NeuralNetwork
 Activation.h
 TrainingAlgorithms.h
 CostFunction.h
 NetworkConfiguration.h
 NeuralNetwork.h
 NeuralNetwork.cpp
)

set(Main
 Main.cpp
)

add_library(NeuralNetwork ${NeuralNetwork})
add_executable(Main ${Main})
