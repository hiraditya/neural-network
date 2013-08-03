#ifndef ANN_NEURON_H
#define ANN_NEURON_H

#include "Axon.h"

#include<graphs/GraphDataTypes.h>

namespace ANN {

  using namespace graphs;
  /**
   * Represents a neuron.
   */
  template<typename NeuronTraitType, typename DendronTraitType>
  class Neuron : public Vertex<Dendron<DendronTraitType>,
                               NeuronTraitType> {
    typedef Vertex<Dendron<DendronTraitType>, NeuronTraitType>
            NeuronBaseType;
    typedef Dendron<DendronTraitType> DendronBaseType;
    public:
      typedef typename NeuronBaseType::EdgesType DendronsType;
      // An empty neuron.
      Neuron(NeuronTraitType n)
        : NeuronBaseType(n)
      { }
      // Neuron with connections.
      Neuron(DendronBaseType d, NeuronTraitType n)
        : NeuronBaseType(d, n)
      { }

      void SetNeuronTrait(const NeuronTraitType& n)
      {
        SetWeight(n);
      }

      void AddDendron(DendronBaseType* d)
      {
        AddEdge(d);
      }

      void RemoveDendron(DendronBaseType* d)
      {
        NeuronBaseType::RemoveEdge(d);
      }

      bool HasDendron(DendronBaseType* d)
      {
        return NeuronBaseType::HasEdge(d);
      }

      const DendronsType& GetDendrons() const
      {
        return NeuronBaseType::GetEdges();
      }

      const NeuronTraitType& GetNeuronTrait() const
      {
        return NeuronBaseType::GetWeight();
      }

      void print(std::ostream& os,
           const Neuron<NeuronTraitType, DendronTraitType>& n) const
      {
        os << n;
      }

      void dump() const
      {
        print(dbgs(), *this);
      }

      friend std::ostream& operator<<(std::ostream& os,
          const Neuron<NeuronTraitType, DendronTraitType>& n)
      {
        os<<static_cast<const NeuronBaseType&>(n);
        return os;
      }
  };
} // namespace ANN
#endif // ANN_NEURON_H
