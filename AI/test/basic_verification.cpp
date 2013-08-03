#include<Neuron.h>
#include<iostream>
#include<cstdlib>
#include<list>
#include<map>
#include<vector>
#include<algorithm>
#include<cassert>

using namespace ANN;
using namespace graphs;

//using DendronTraitType = TraitType;
//using NeuronTraitType = TraitType;
/**
 * This program sets up the basic connections in neurons and
 * dendrons.
 * It also verifies the connections a little bit.
 */
int main()
{ 
  typedef Dendron<TraitType> DendronType;
  typedef Neuron<TraitType, TraitType> NeuronType;
  std::vector<DendronType> Dendrons;
  std::vector<NeuronType> Neurons;
  std::map<IDType, std::pair<NeuronType*, NeuronType*> > DendronMap;
  int total = 2;
  for (int i = 0; i < total; ++i) {
    int rnd1 = std::rand();
    int rnd2 = std::rand();
    NeuronType n1(TraitType(i, rnd1));
    NeuronType n2(TraitType(100+i, rnd2));
    Neurons.push_back(n1);
    Neurons.push_back(n2);
    std::cout<<"\nEnd For Loop InsertNeuron";
  }
  for (int i = 0; i < 8*total; ++i) {
    int rnd1 = std::rand()%Neurons.size();
    int rnd2 = std::rand()%Neurons.size();
    //std::cout << "\nRandon Numbers:" << rnd1 << rnd2;
    Dendron<TraitType> d(Neurons[rnd1],
           Neurons[rnd2], TraitType(1000+i, rnd1));
    Dendrons.push_back(d);
    DendronMap[d.GetId()] = std::make_pair(&Neurons[rnd1], &Neurons[rnd2]);
    std::cout<<"\nEnd For Loop InsertDendron";
  }

  std::cout<< "\nPrinting the dendrons:";
  for (auto i = Dendrons.begin(); i != Dendrons.end(); ++i) {
    std::cout << "\n" << *i;
    auto conn = DendronMap[i->GetId()];
    NeuronType* in = conn.first;
    NeuronType* out = conn.second;
    auto in_dends = in->GetDendrons();
    auto out_dends = out->GetDendrons();
    auto in_it = std::find(in_dends.begin(), in_dends.end(),
                           i->GetId());
    auto out_it = std::find(out_dends.begin(), out_dends.end(),
                           i->GetId());
    assert((in_it != in_dends.end()) &&
           (out_it != out_dends.end()));
  }

  std::cout<< "\nPrinting the neurons:";
  for (auto i = Neurons.begin(); i != Neurons.end(); ++i) {
    std::cout << "\n" << *i;
  }
  return 0;
}
