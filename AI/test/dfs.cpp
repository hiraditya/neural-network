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
 * It also performs a depth first on all the neurons
 * and print the graph in dot format. 
 */
int main()
{ 
  typedef Dendron<TraitType> DendronType;
  typedef Neuron<TraitType, TraitType> NeuronType;
  std::vector<DendronType> Dendrons;
  std::vector<NeuronType> Neurons;
  int total = 2;
  for (int i = 0; i < total; ++i) {
    int rnd1 = std::rand();
    int rnd2 = std::rand();
    NeuronType n1(TraitType(i, rnd1));
    NeuronType n2(TraitType(100+i, rnd2));
    Neurons.push_back(n1);
    Neurons.push_back(n2);
  }
  for (int i = 0; i < 8*total; ++i) {
    int rnd1 = std::rand()%Neurons.size();
    int rnd2 = std::rand()%Neurons.size();
    Dendron<TraitType> d(Neurons[rnd1],
           Neurons[rnd2], TraitType(1000+i, rnd1));
    Dendrons.push_back(d);
  }
  /*for (auto i = Neurons.begin(); i != Neurons.end(); ++i) {
    auto n_id = i->GetId();
    std::for_each(i->GetDendrons().begin(), i->GetDendrons().end(),
                  [n_id](const IDType d_id){
                    std::cout<<"\nn_id"<<n_id<<"->"<<"d_id"<<d_id;
                  });
  }*/
  std::cout<<"digraph network {";
  for (auto i = Dendrons.begin(); i != Dendrons.end(); ++i) {
    auto d_id = i->GetId();
    auto in_id = i->GetInNeuron()->GetId();
    auto out_id = i->GetOutNeuron()->GetId();
    std::cout<<"\n\tn_id"<<in_id<<"->"<<"d_id"<<d_id<<';';
    std::cout<<"\n\td_id"<<d_id<<"->"<<"n_id"<<out_id<<';';
  }
  std::cout<<"\n}";
  return 0;
}
