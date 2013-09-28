#include<NeuralNetwork.h>
#include<iostream>

using namespace ANN;
using namespace utilities;
// Sets up the single layer feed forward network
// and evaluates the output for a set of sample inputs.
int main() {
  const int ip_size = 6;
  auto NN = CreateSLFFN<SigmoidAct<DendronWeightType>>(ip_size);
  std::vector<float> SampleSpace{0, 1};
  for (unsigned i = 0; i < 2; ++i) {
    std::vector<float> RS = GetRandomizedSet(SampleSpace, ip_size-1);
    // The last input is the bias.
    RS.push_back(-1);
    //DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RS);
    // dbgs() << "\nOutput: " << NN.GetOutput(RS));
    NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
    auto op = NN.GetOutput(RS);
    //NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
    //using NNType = decltype(NN);
    Trainer<decltype(NN), GradientDescent>T(NN);
    T.TrainNetwork(RS, op);
    NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
  }
  return 0;
}
