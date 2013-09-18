#ifndef ANN_NEURAL_NETWORK_H
#define ANN_NEURAL_NETWORK_H

#include<Debug.h>
#include<RandomGenerator.h>

#include<vector>
#include<list>
#include<algorithm>
#include<cassert>
#include<numeric>
#include<cmath>


namespace ANN {

  typedef float WeightType;
  typedef WeightType DendronWeightType;
  typedef WeightType NeuronWeightType;
  typedef unsigned LevelType;
  typedef unsigned IdType;

  static WeightType FPDelta = 0.000001;
  typedef std::vector<WeightType> InputVector;
  struct NeuronType;

  template<typename WeightType>
  inline bool Valid(WeightType& w) {
    return w >= 0;
  }

  // @todo Reimplement this.
  IdType GetNewId() {
    static IdType ID = 0;
    return ID++;
  }

  enum Direction {
    InvalidDirection = 0,
    Inwards,
    Outwards
  };

  // Represents one denron. Each dendron must be
  // connected to two neurons.
  //template<typename DendronWeightType>
  struct DendronType {
    IdType Id;
    NeuronType* In;
    NeuronType* Out;
    DendronWeightType W;

    // in -----dendron--------> out
    // output of 'in' neuron feeds to the input of 'out' neuron
    DendronType(IdType i, NeuronType* in, NeuronType* out,
                DendronWeightType w)
      : Id(i), In(in), Out(out), W(w){
        assert(In && Out);
    }

    bool operator==(const DendronType& d) const {
      return (In == d.In) && (Out == d.Out) &&
             (std::fabs(W - d.W) < FPDelta);
    }

    bool operator!=(const DendronType& d) const {
      return !(*this == d);
    }
  };

  typedef std::vector<DendronType*> DendronsRefType;
  // Represents one neuron. Each neuron may have multiple
  // incoming/outgoing connections.
  //template<typename NeuronWeightType>
  struct NeuronType {
    IdType Id;
    DendronsRefType Ins;
    DendronsRefType Outs;
    NeuronWeightType W;

    NeuronType(IdType i, NeuronWeightType w)
      : Id(i), W(w)
    { }
    NeuronType(IdType i, DendronType* in, DendronType* out, NeuronWeightType w)
      : Id(i), W(w) {
      Ins.push_back(in);
      Outs.push_back(out);
    }

    bool operator==(const NeuronType& n) const {
      return (n.Ins == Ins) && (n.Outs == Outs) &&
             (std::fabs(n.W - W) < FPDelta);
    }

    bool operator!=(const NeuronType& n) const {
      return !(*this == n);
    }

    // in ----> (this Neuron)
    // (this Neuron) ----> out
    // Direction is w.r.t. this neuron.
    void Connect(DendronType* d, Direction dir) {
      assert(d);
      assert(dir != Direction::InvalidDirection);
      // Self feedback is not supported yet.
      assert(std::find(Ins.begin(), Ins.end(), d) == Ins.end());
      assert(std::find(Outs.begin(), Outs.end(), d) == Outs.end());
      if (dir == Direction::Inwards) {
        assert(d->Out == this);
        Ins.push_back(d);
      } else { // Outwards
        assert(d->In == this);
        Outs.push_back(d);
      }
    }

    // --->in--->(this Neuron)--->out--->
    // this Neuron (In) = in
    // this Neuron (Out) = out
    //
    void Connect(DendronType* in, DendronType* out) {
      Connect(in, Direction::Inwards);
      Connect(out, Direction::Outwards);
    }

    void SetWeight(NeuronWeightType w) {
      assert(Valid(w));
      W = w;
    }

    // Removes the entry of Dendron \d from the connections.
    void Disconnect(DendronType* d, Direction dir) {
      assert(dir != Direction::InvalidDirection);
      // Assume that incoming dendron is to be removed.
      DendronsRefType* ToBe = &Ins;
      if (dir == Direction::Outwards)
        ToBe = &Outs;
      DendronsRefType::iterator it = std::find(ToBe->begin(), ToBe->end(), d);
      assert(it != ToBe->end());
      ToBe->erase(it);
    }
    void Disconnect(DendronType* d) {
      Disconnect(d, Direction::Inwards);
      Disconnect(d, Direction::Outwards);
    }

    Direction IsConnected(DendronType* d) {
      DendronsRefType::iterator it = std::find(Ins.begin(), Ins.end(), d);
      if (it != Ins.end())
        return Direction::Inwards;
      it = std::find(Outs.begin(), Outs.end(), d);
      if (it != Outs.end())
        return Direction::Outwards;
      return Direction::InvalidDirection;
    }

    DendronWeightType EvalOp(InputVector& ip) {
      assert(ip.size() == Ins.size());
      auto d = Ins.begin();
      auto i = ip.begin();
      DendronWeightType Sum = 0;
      while(d != Ins.end()) {
        Sum += (*i) * (*d)->W;
        ++d;
        ++i;
      }
      return Sum;
    }

    template<typename Stream>
    void Print(Stream& s) {
      s << "\n\tn" << Id
        << " [ label = \"n" << Id << "(W:" << W << ")\"];";
    }
  };


  typedef std::list<NeuronType> NeuronsType;
  typedef std::list<NeuronType*> NeuronsRefType;
  typedef std::list<DendronType> DendronsType;
  /**
   *   ,--In--Cen---  -----
   *  N --In--Cen---  ----- Out
   *   `--In--Cen---  -----
   *  N is the root-neuron (kind of placeholder) and all the pseudo-edges
   *  coming out of N are Input Dendrons.
   *  This makes it easy to evaluate the network uniformly.
   */
  class NeuralNetwork {
    NeuronsType Neurons;
    DendronsType Dendrons;
    // Root is always the first entry in Neurons.
    NeuronType* RootNeuron;
    public:
      // Empty
      NeuralNetwork()
        : RootNeuron(NULL)
      { }

      // Initializes the network with just one neuron.
      NeuralNetwork(NeuronType& n) {
        Neurons.push_back(n);
        RootNeuron = &*Neurons.begin();
      }

      NeuronsType::iterator CreateRoot() {
        assert(RootNeuron == NULL);
        CreateNeuron();
        RootNeuron = &*Neurons.begin();
        return Neurons.begin();
      }
      /*/ If the network size is known beforehand.
      void Resize(unsigned numNeurons, unsigned numDendrons) {
        Neurons.resize(numNeurons);
        Dendrons.resize(numDendrons);
      }*/

      // Root is always the first entry.
      NeuronsType::iterator GetRoot() {
        assert(RootNeuron);
        return Neurons.begin();
      }

      // Use the iterator, don't reuse it.
      // It might have been invalidated.
      NeuronsType::iterator CreateNeuron(NeuronWeightType w 
                                         = NeuronWeightType(0)) {
        NeuronType n(GetNewId(), w);
        return AddNeuron(n);
      }

      // create a dendron that connects
      // i1 ------> i2
      // Use the iterator, don't reuse it.
      // It might have been invalidated.
      DendronsType::iterator CreateDendron(NeuronsType::iterator i1,
                                           NeuronsType::iterator i2,
                         DendronWeightType w = DendronWeightType(0)) {
        DendronType d(GetNewId(), &*i1, &*i2, w);
        return AddDendron(d);
      }
      // @warning: This should be used in conjunction with
      // AddDendron, because in a neural-network neurons
      // are always connected to some other neuron(s).
      NeuronsType::iterator AddNeuron(NeuronType& n) {
        Neurons.push_back(n);
        return --Neurons.end();
      }

      // @warning: This function should never be used from outside.
      // Every dendron (like an edge in the graph) must be connected
      // with a neuron.
      DendronsType::iterator AddDendron(DendronType& d) {
        Dendrons.push_back(d);
        return --Dendrons.end();
      }

      // This function is really intrusive and should be used carefully.
      void Connect(NeuronType& n, DendronType& d, Direction direction) {
        if (direction == Inwards) {
          n.Ins.push_back(&d);
        } else { // Outwards
          n.Outs.push_back(&d);
        }
      }

      void Connect(NeuronType& n, DendronType& d1, DendronType& d2) {
        n.Connect(&d1, &d2);
      }

      // Create a new dendron with n1 as output, n2 as input.
      // n1 ---------> n2
      void Connect(const NeuronType& n1, const NeuronType& n2,
                   DendronWeightType w) {
        auto it1 = std::find(Neurons.begin(), Neurons.end(), n1);
        auto it2 = std::find(Neurons.begin(), Neurons.end(), n2);
        Connect(it1, it2, w);
      }

      // Create a new dendron with i1 as output, i2 as input.
      // i1 ---------> i2
      // Enter the connection in each neuron.
      void Connect(NeuronsType::iterator i1, NeuronsType::iterator i2,
          DendronWeightType w) {
        assert(i1 != Neurons.end());
        assert(i2 != Neurons.end());
        assert(i1 != i2);
        auto dp = CreateDendron(i1, i2, w);
        // d is the output of i1
        i1->Connect(&*dp, Direction::Outwards);
        // d is the input of i2
        i2->Connect(&*dp, Direction::Inwards);
      }

      void Remove(NeuronType& n) {
        assert(n != *RootNeuron);
        NeuronsType::iterator it = std::find(Neurons.begin(), Neurons.end(), n);
        assert(it != Neurons.end());
        for (auto ins = n.Ins.begin(); ins != n.Ins.end(); ++ins) {
          Remove(**ins);
        }
        for (auto outs = n.Outs.begin(); outs != n.Outs.end(); ++outs) {
          Remove(**outs);
        }
        Neurons.erase(it);
      }

      void Remove(DendronType& d) {
        DendronsType::iterator it = std::find(Dendrons.begin(), Dendrons.end(), d);
        assert(it != Dendrons.end());
        // d.In--->Out-In---->dendron---->Out-In---->d.Out
        d.In->Disconnect(&d, Direction::Outwards);
        d.Out->Disconnect(&d, Direction::Inwards);
        Dendrons.erase(it);
      }

      template<typename Stream>
      void PrintNetwork(Stream& s, std::string Title) {
        s << "digraph " << Title << "{\n";
        PrintNeurons(s);
        s << "};";
      }

      template<typename Stream>
      void PrintNeurons(Stream& s) {
        for(auto ni = Neurons.begin(); ni != Neurons.end(); ++ni)
          ni->Print(s);
      }

      // @todo: Put innovation number as well.
      template<typename Stream>
      void PrintDendron(Stream& s, DendronType& d) {
        s << "\n\tn" << d.In->Id
          << "->n" << d.Out->Id
          << "[ label = \" d" << d.Id << " (" << d.W<< ")\"];";
      }

      // Breadth First traversal of network
      template<typename Stream>
      void PrintConnections(NeuronType& root, Stream& s) {
        for (auto dp = root.Outs.begin(); dp != root.Outs.end();
            ++dp) {
          PrintDendron(s, **dp);
          // This should work now because self-feedback is not supported.
          PrintConnections(*(*dp)->Out, s);
        }
      }
      template<typename Stream>
      void PrintNNDigraph(NeuronType& root, Stream& s,
                           const std::string& Name= "") {
        s << "digraph " << Name <<  " {\n";
        PrintConnections(root, s);
        s << "\n}\n";
      }
  };

  // Single Layer Feed Forward network.
  // ,----IN ----
  // N----IN ---- Out
  // `----IN ----
  // @param NumNeurons = Number of input layer neurons.
  NeuralNetwork CreateSLFFN(unsigned NumNeurons) {
    using namespace utilities;
    RNG rnd(0, 1);
    NeuralNetwork nn;
    auto root = nn.CreateRoot();
    auto out = nn.CreateNeuron(rnd.Get());
    for (unsigned i = 0; i < NumNeurons; ++i) {
      auto in = nn.CreateNeuron(rnd.Get());
      nn.Connect(root, in, rnd.Get());
      nn.Connect(in, out, rnd.Get());
    }
    return nn;
  }
  // Two Layer Feed Forward network
  // ,---- IN ---- IN
  // ,---- IN ---- IN
  // N-----IN ---- IN-- Out
  // `---- IN ---- IN//
  // `---- IN ---- IN/
  NeuralNetwork CreateTLFFN(unsigned NumNeurons) {
    using namespace utilities;
    RNG rnd(0, 1);
    NeuralNetwork nn;
    auto root = nn.CreateRoot();
    auto out = nn.CreateNeuron(rnd.Get());
    for (unsigned i = 0; i < NumNeurons; ++i) {
      auto in1 = nn.CreateNeuron(rnd.Get());
      auto in2 = nn.CreateNeuron(rnd.Get());
      nn.Connect(root, in1, rnd.Get());
      nn.Connect(in1, in2, rnd.Get());
      nn.Connect(in2, out, rnd.Get());
    }
    return nn;
  }
} // namespace ANN

#endif // ANN_NEURAL_NETWORK_H
