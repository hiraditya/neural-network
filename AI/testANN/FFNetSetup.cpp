#include<NeuralNetwork.h>
#include<iostream>

using namespace ANN;

int main() {
  const int ip_size = 4;
  //auto NN = CreateSLFFN<LinearAct<DendronWeightType>>(ip_size);
  //NN.PrintNNDigraph(*NN.GetRoot(), std::cout);

  auto TL = CreateTLFFN<LinearAct<DendronWeightType>>(ip_size);
  TL.PrintNNDigraph(*TL.GetRoot(), std::cout);
  return 0;
}
