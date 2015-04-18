/**
 * This program implements XOR problem
 *
 */

#include<utilities/Debug.h>
#include<Neuron.h>
#include<iostream>
#include<cstdlib>
#include<list>
#include<map>
#include<vector>
#include<algorithm>
#include<string>
#include<sstream>
#include<cassert>
#include<limits>
#include<random>
#include<cmath>

using namespace ANN;
using namespace graphs;
enum Function {
  AND,
  OR,
  XOR
};

bool GetOp(bool i1, bool i2, Function func)
{
  switch (func) {
    case AND: return i1 && i2;
    case OR: return i1 || i2;
    case XOR: return i1 ^ i2;
    default: assert(0);
  }
}

typedef Dendron<TraitType> DendronType;
typedef Neuron<TraitType, TraitType> NeuronType;
typedef float WeightType;

bool ValidateNetwork(const std::vector<NeuronType>& Neurons,
                     const std::vector<DendronType>& Dendrons,
                     Function fn) {
  std::vector<std::pair<int, int> > input{{0,0}, {0,1}, {1,0}, {1,1}};
  bool trained = false;
  for (auto it = input.begin(); it != input.end(); ++it) {
    auto i1 = it->first, i2 = it->second;
    auto n0_wt = Neurons[0].GetNeuronTrait().Weight;
    auto n1_wt = Neurons[1].GetNeuronTrait().Weight;
    auto n2_wt = Neurons[2].GetNeuronTrait().Weight;
    auto d0_wt = Dendrons[0].GetDendronTrait().Weight;
    auto d1_wt = Dendrons[1].GetDendronTrait().Weight;
    auto d2_wt = Dendrons[2].GetDendronTrait().Weight;
    WeightType edge_ip = i1*n0_wt*d0_wt + i2*n1_wt*d1_wt + n2_wt*d2_wt;
    WeightType delta = edge_ip - Neurons[3].GetNeuronTrait().Weight;
    std::cout << "\ni1:" << i1 << " i2:" << i2
              << " delta:" << delta << "\n";
    bool op = GetOp(i1, i2, fn);
    trained = (op && delta > 0) || (!op && delta < 0);
    if (!trained)
      return false;
  }
  return true;
}

int main()
{
  std::vector<DendronType> Dendrons;
  std::vector<NeuronType> Neurons;
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, 1);
  Function fn = Function::OR;

  WeightType rnd1 = dist(e2);
  WeightType rnd2 = dist(e2);
  WeightType rnd3 = -dist(e2); // bias
  WeightType rnd4 = 1; // dist(e2);
  int i = 0;
  // create input neurons
  NeuronType in1(TraitType(rnd1, i++));
  NeuronType in2(TraitType(rnd2, i++));
  NeuronType bias(TraitType(rnd3, i++));
  NeuronType output(TraitType(rnd4, i++));

  // store the neurons
  Neurons.push_back(in1);
  Neurons.push_back(in2);
  Neurons.push_back(bias);
  Neurons.push_back(output);

  WeightType rnd5 = dist(e2);
  WeightType rnd6 = dist(e2);
  WeightType rnd7 = dist(e2);

  Dendron<TraitType> d1(Neurons[0], Neurons[3], TraitType(rnd5, i++));
  Dendron<TraitType> d2(Neurons[1], Neurons[3], TraitType(rnd6, i++));
  Dendron<TraitType> d3(Neurons[2], Neurons[3], TraitType(rnd7, i++));
  Dendrons.push_back(d1);
  Dendrons.push_back(d2);
  Dendrons.push_back(d3);

  float alpha = -0.001;
  float alpha2 = +0.001;
  int trainIter = 0;
  int all_trained = 0;
  std::vector<int> inputs = {0 , 1};
  while (/*trainIter < 100000 &&*/ all_trained < 50) {
    ++trainIter;
    bool trained = false;
    int max_iter = 10;
    int num_iter = 0;
    while (!trained) {
      int rnd8 = std::lround(dist(e2)*100)%2;
      int rnd9 = std::lround(dist(e2)*100)%2;
      auto n0_wt = Neurons[0].GetNeuronTrait().Weight;
      auto n1_wt = Neurons[1].GetNeuronTrait().Weight;
      auto n2_wt = Neurons[2].GetNeuronTrait().Weight;
      auto d0_wt = Dendrons[0].GetDendronTrait().Weight;
      auto d1_wt = Dendrons[1].GetDendronTrait().Weight;
      auto d2_wt = Dendrons[2].GetDendronTrait().Weight;

      WeightType edge_ip = rnd8*n0_wt*d0_wt + rnd9*n1_wt*d1_wt + n2_wt*d2_wt;
      WeightType delta = edge_ip - Neurons[3].GetNeuronTrait().Weight;
      std::cout << "\ni1:" << rnd8 << " i2:" << rnd9
                << " delta:" << delta << "\n";
      // Evaluation Function
      bool op = GetOp(rnd8, rnd9, fn);
      trained = (op && delta > 0) || (!op && delta < 0);
      std::stringstream strm;
      if (!trained && num_iter < max_iter) {
        ++num_iter;
        all_trained = 0;
        n0_wt = n0_wt + alpha2*delta;
        n1_wt = n1_wt + alpha2*delta;
        n2_wt = n2_wt + alpha2*delta;
        Neurons[0].SetNeuronTrait(TraitType(n0_wt, i++));
        Neurons[1].SetNeuronTrait(TraitType(n1_wt, i++));
        Neurons[2].SetNeuronTrait(TraitType(n2_wt, i++));

        d0_wt = d0_wt + alpha2*delta;
        d1_wt = d1_wt + alpha2*delta;
        d2_wt = d2_wt + alpha2*delta;
        Dendrons[0].SetDendronTrait(TraitType(d0_wt, i++));
        Dendrons[1].SetDendronTrait(TraitType(d1_wt, i++));
        Dendrons[2].SetDendronTrait(TraitType(d2_wt, i++));
        //strm << "Untrained" << trainIter;
        //DEBUG0(dbgs() << "\nUntrained" << num_iter);
        //PrintNetwork(std::cout, Dendrons.begin(), Dendrons.end(),
        //             strm.str());
      } else {
        if (num_iter < max_iter) {
          ++all_trained;
          n0_wt = n0_wt + alpha*delta;
          n1_wt = n1_wt + alpha*delta;
          n2_wt = n2_wt + alpha*delta;
          Neurons[0].SetNeuronTrait(TraitType(n0_wt, i++));
          Neurons[1].SetNeuronTrait(TraitType(n1_wt, i++));
          Neurons[2].SetNeuronTrait(TraitType(n2_wt, i++));
        
          d0_wt = d0_wt + alpha*delta;
          d1_wt = d1_wt + alpha*delta;
          d2_wt = d2_wt + alpha*delta;
          Dendrons[0].SetDendronTrait(TraitType(d0_wt, i++));
          Dendrons[1].SetDendronTrait(TraitType(d1_wt, i++));
          Dendrons[2].SetDendronTrait(TraitType(d2_wt, i++));
          strm << "Trained" << trainIter;
          //DEBUG0(dbgs() << "\nTrained" << num_iter);
          PrintNetwork(std::cout, Dendrons.begin(), Dendrons.end(),
                       strm.str());
        }
        trained = true;
      }
    }
  }
  if (ValidateNetwork(Neurons, Dendrons, fn))
    PrintNetwork(std::cout, Dendrons.begin(), Dendrons.end(), "FinalOp");
  else
    std::cout << "\nUntrained";
  return 0;
}
