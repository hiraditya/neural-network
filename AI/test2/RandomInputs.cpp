#include<NeuralNetwork.h>
#include<iostream>

using namespace ANN;
using namespace utilities;
// Sets up the single layer feed forward network
// and evaluates the output for a set of sample inputs.
int main() {
  auto NN = CreateSLFFN(3);
  std::vector<float> SampleSpace{0, 1};
  for (unsigned i = 0; i < 2; ++i) {
    std::vector<float> RS = GetRandomizedSet(SampleSpace, 2);
    // The last input is the bias.
    RS.push_back(-1);
    DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RS));
    NN.GetOutput(RS);
    NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
  }
  return 0;
}
