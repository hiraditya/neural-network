#include<NeuralNetwork.h>
#include<iostream>

using namespace ANN;
// Sets up the single layer feed forward network
// and evaluates the output for one sample input.
int main() {
  auto NN = CreateSLFFN(3);
  std::vector<float> v{0, 1, -1};
  //NeuronType* Root = &*NN.GetRoot();
  //NN.PrintNNDigraph(*Root, std::cout);
  using namespace utilities;
  //std::cout << "\nFinal Output: " << NN.GetOutput(v);
  NN.GetOutput(v);
  NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
  return 0;
}
