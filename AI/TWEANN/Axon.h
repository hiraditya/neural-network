#ifndef ANN_AXON_H
#define ANN_AXON_H

#include<graphs/GraphDataTypes.h>
#include<utilities/Printer.h>

#include<iostream>
#include<set>
#include<algorithm>

namespace ANN {
  using namespace graphs;
  typedef float WeightType;
  typedef int EvolutionType;
  template<typename NeuronTraitType, typename DendronTraitType>
  class Neuron;

  struct TraitType {
    WeightType Weight;
    EvolutionType Evolution;
    TraitType(WeightType w, EvolutionType e)
      : Weight(w), Evolution(e)
    { }

    bool operator<(const TraitType& other) const {
      return Weight < other.Weight ? true : Evolution < other.Evolution;
    }

    friend
    std::ostream& operator<<(std::ostream& os,
                          const TraitType& t)
    {
      if (printer::YAML) {
        os << "\n\t- Weight: " << t.Weight
           << "\n\t- Evolution: " << t.Evolution;
      } else { // CSV
        os << "(W:" << t.Weight << ", E:" << t.Evolution << ')';
      }
      return os;
    }
  };

  /**
   * Dendron represent the in/out connections of a neuron.
   */
  template<typename DendronTraitType>
  class Dendron : public Edge<Neuron<TraitType, DendronTraitType>,
                              DendronTraitType> {
    public:
      typedef TraitType NeuronTraitType;
      typedef Neuron<TraitType, DendronTraitType> NeuronBaseType;
      typedef Edge<Neuron<TraitType, DendronTraitType>,
                   DendronTraitType> DendronBaseType;

      // Construct an dendron when In and Out neurons are known.
      Dendron(NeuronBaseType& In, NeuronBaseType& Out, DendronTraitType W)
        : DendronBaseType(In, Out, W)
      {
        DendronBaseType::InNode->AddEdge(this);
        DendronBaseType::OutNode->AddEdge(this);
      }

      void SetDendronTrait(const DendronTraitType& n)
      {
        DendronBaseType::SetWeight(n);
      }

      const DendronTraitType& GetDendronTrait() const
      {
        return DendronBaseType::GetWeight();
      }

      NeuronBaseType* GetInNeuron()
      {
        return DendronBaseType::GetInNode();
      }

      NeuronBaseType* GetOutNeuron()
      {
        return DendronBaseType::GetOutNode();
      }

      void print(std::ostream& os,
                 const Dendron<DendronTraitType>& d) const
      {
        os << d;
      }

      void dump() const
      {
        print(dbgs(), *this);
      }

      friend
      std::ostream& operator<<(std::ostream& os,
                            const Dendron<DendronTraitType>& d)
      {
        os << static_cast<const DendronBaseType&>(d);
        return os;
      }

      typedef std::pair<IDType, NeuronTraitType> NeuronDetails;

      friend
      bool operator<(const NeuronDetails& n1, const NeuronDetails& n2)
      {
        return n1.first < n2.first;
      }

      template<typename InputIterator>
      friend void PrintNetwork(std::ostream& os,
             const InputIterator begin, const InputIterator end,
             const std::string& name)
      {
        std::set<NeuronDetails> NeuronData;
        os << "digraph " << name << " {\nlandscape=true;";
        for (auto i = begin; i != end; ++i) {
          auto in_id = i->GetInNeuron()->GetId();
          auto out_id = i->GetOutNeuron()->GetId();
          auto in_tr = i->GetInNeuron()->GetNeuronTrait();
          auto out_tr = i->GetOutNeuron()->GetNeuronTrait();
          NeuronData.insert(std::make_pair(in_id, in_tr));
          NeuronData.insert(std::make_pair(out_id, out_tr));
        }
        for (auto i = NeuronData.begin(); i != NeuronData.end(); ++i) {
          os << "\n\tn" << i->first << " [ label = \"n" << i->first
            << i->second << "\"];";
        }
        for (auto i = begin; i != end; ++i) {
          auto d_id = i->GetId();
          auto in_id = i->GetInNeuron()->GetId();
          auto out_id = i->GetOutNeuron()->GetId();
          os << "\n\tn" << in_id << "->" << "n" << out_id
             <<" [ label = \"d" << d_id << *i << "\"];";
        }
        os<<"\n};\n";
      }

  };

  /**
   * Represents an axon. Axon has been represented as a separate entity
   * from neuron.
   */
  template<typename DendronTraitType>
  class Axon : public Dendron<DendronTraitType>
  {
    public:
      typedef Dendron<DendronTraitType> DendronType;
      typedef typename DendronType::NeuronBaseType NeuronBaseType;
      typedef typename DendronType::DendronBaseType DendronBaseType;
      // Construct an axon when In and Out neurons are known.
      Axon(NeuronBaseType& In, NeuronBaseType& Out, DendronTraitType W)
         : DendronBaseType(In, Out, W)
      {
        DendronBaseType::InNode->AddEdge(this);
        DendronBaseType::OutNode->AddEdge(this);
      }

      void print(std::ostream& os,
                 const Dendron<DendronTraitType>& d) const
      {
        os << d;
      }

      void dump() const
      {
        print(dbgs(), *this);
      }

      friend std::ostream& operator<<(std::ostream& os,
                            const Dendron<DendronTraitType>& d)
      {
        os << static_cast<const DendronBaseType&>(d);
        return os;
      }
  };

} // namespace ANN

#endif // ANN_AXON_H
