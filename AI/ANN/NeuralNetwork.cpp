#include"NeuralNetwork.h"

// This function should be used to set negative weight
// of bias neurons.
using namespace ANN;

void DendronType::SetBiasWeight(DendronWeightType w) {
  assert(IsRootNeuron(In->W));
  W = w;
}
