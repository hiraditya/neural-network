#include<NeuralNetwork.h>
#include<iostream>

using namespace ANN;

int main() {
  ANN::NeuralNetwork nn;
  auto NN = CreateSLFFN(5);
  NN.PrintNNDigraph(*NN.GetRoot(), std::cout);

  auto TL = CreateTLFFN(5);
  TL.PrintNNDigraph(*TL.GetRoot(), std::cout);
  return 0;
}
