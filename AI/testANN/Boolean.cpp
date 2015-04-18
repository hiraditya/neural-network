#include"NeuralNetwork.h"
#include"TypeConversions.h"

#include<iostream>

using namespace ANN;
// Sets up the single layer feed forward network
// and evaluates the output for one sample input.
int main() {
  const int ip_size = 3;
  auto NN = CreateSLFFN<LinearAct<DendronWeightType>>(ip_size);
  using namespace utilities;
  std::vector<bool> RS = GetRandomizedSet(BooleanSampleSpace, ip_size-1);
  std::vector<float> RSF = BoolsToFloats(RS);
  // The last input is the bias.
  RSF.push_back(-1);
  DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RSF));
  //NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
  NN.GetOutput(RSF);
  //std::cout << "\nFinal Output: " << NN.GetOutput(v);
  NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
  return 0;
}
