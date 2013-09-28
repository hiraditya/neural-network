cmake_minimum_required(VERSION 2.8)

set(NeuralNetwork
 Activation.h
 TrainingAlgorithms.h
 NeuralNetwork.h
 NeuralNetwork.cpp
)


add_library(NeuralNetwork ${NeuralNetwork})
