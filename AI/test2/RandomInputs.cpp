#include<NeuralNetwork.h>
#include<TypeConversions.h>

#include<iostream>

using namespace ANN;
using namespace utilities;
// Sets up the single layer feed forward network
// and evaluates the output for a set of sample inputs.
int main() {
  const int ip_size = 3;
  // Create a single layer feed forward neural network.
  auto NN = CreateSLFFN<LinearAct<DendronWeightType>>(ip_size);

  // Choose the training algorithm.
  Trainer<decltype(NN), GradientDescent>T(NN);

  // Validation of the output.
  typedef ValidateOutput<decltype(NN), std::vector<bool>, bool>
          ValidatorType;
  ValidatorType Validator(NN);
  typedef BoolOr CostFunction;

  for (unsigned i = 0; i < 20;) {
    std::vector<bool> RS = GetRandomizedSet(BooleanSampleSpace, ip_size-1);
    std::vector<float> RSF = BoolsToFloats(RS);
    // The last input is the bias.
    RSF.push_back(-1);
    DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RSF));
    //NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
    auto op = NN.GetOutput(RSF);
    // Is the output same as desired output?
    if (!Validator.Validate(CostFunction(), RS, FloatToBool(op))) {
      DEBUG0(dbgs() << "\tLearning (" << op << ", " << FloatToBool(op) << ")");
      // No => Train
      T.TrainNetwork(RSF, op);
      i = 0;
    } else {
      ++i; // Increment trained counter.
      DEBUG0(dbgs() << "\tTrained (" << op << ", " << FloatToBool(op) << ")");
    }
  }
  // Once the training has been done, test the network with
  // a large set of inputs in the sample space, possibly
  // covering the complete sample space.
  DEBUG0(dbgs() << "\nPrinting after training");
  for (unsigned i = 0; i < 10; ++i) {
    std::vector<bool> RS = GetRandomizedSet(BooleanSampleSpace, ip_size-1);
    std::vector<float> RSF = BoolsToFloats(RS);
    // The last input is the bias.
    RSF.push_back(-1);
    auto op = NN.GetOutput(RSF);
    DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RSF));
    if (Validator.Validate(CostFunction(), RS, FloatToBool(op)))
      DEBUG0(dbgs() << "\tTrained: " << op);
    else
      DEBUG0(dbgs() << "\tUnTrained: " << op);
  }
  NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
  return 0;
}
