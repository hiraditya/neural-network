#ifndef ANN_NEURAL_NETWORK_H
#define ANN_NEURAL_NETWORK_H

#include "Activation.h"
#include "TypeConversions.h"
#include "CostFunction.h"

#include <Debug.h>
#include <RandomGenerator.h>
#include <Distance.h>

#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <cmath>
/**
 * TODO: Serialization
 * Abstract breadth first traversal from train-network and
 * depth first traversal from print-network.
 * Facility to report statistics. Helper class that one could
 * call at various stages to log the state of neural network.
 * Ability to parse parameters from command line.
 */

namespace ANN {

  typedef float WeightType;
  typedef WeightType DendronWeightType;
  typedef WeightType NeuronWeightType;
  typedef unsigned LevelType;
  typedef unsigned IdType;
  typedef std::vector<WeightType> InputVector;
  struct NeuronType;

  template<typename WeightType>
  inline bool Valid(WeightType& w) {
    return true;
  }

  inline bool IsRootNeuron(IdType Id) {
    return Id == 0;
  }
  inline bool IsOutNeuron(IdType Id) {
    return Id == 1;
  }
  inline bool IsBiasNeuron(IdType Id) {
    return Id == 2;
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

  utilities::FloatingPointDistance DendronDist;
  utilities::FloatingPointDistance NeuronDist;
  // Represents one denron. Each dendron must be
  // connected to two neurons.
  //template<typename DendronWeightType>
  struct DendronType {
    IdType Id;
    NeuronType* In;
    NeuronType* Out;
    DendronWeightType W;
    // delta of previous weight update.
    DendronWeightType dW;

    // in -----dendron--------> out
    // output of 'in' neuron feeds to the input of 'out' neuron
    DendronType(IdType i, NeuronType* in, NeuronType* out,
                DendronWeightType w)
      : Id(i), In(in), Out(out), W(w), dW(DendronWeightType(0)) {
        assert(In && Out);
    }

    void SetWeight(DendronWeightType w) {
      assert(Valid(w));
      W = w;
    }

    // This function should be used to set negative weight
    // of bias neurons.
    void SetBiasWeight(DendronWeightType w);

    bool operator==(const DendronType& d) const {
      return (In == d.In) && (Out == d.Out) &&
              DendronDist.CloseEnough(d.W, W);
    }

    bool operator!=(const DendronType& d) const {
      return !(*this == d);
    }
  };

  typedef std::list<DendronType*> DendronsRefListType;
  typedef std::set<DendronType*> DendronsRefSetType;
  typedef std::vector<DendronType*> DendronsRefVecType;
  // Represents one neuron. Each neuron may have multiple
  // incoming/outgoing connections.
  //template<typename NeuronWeightType>
  struct NeuronType {
    typedef DendronType DendronType_t;
    IdType Id;
    IdType LayerNum;
    NeuronWeightType W; // This is also called as the signal
    NeuronWeightType dW; // error = Output - desired_op
    NeuronWeightType Output; // Output = non-linearity(W)
    bool EvalSigCalled;
    DendronsRefVecType Ins;
    DendronsRefVecType Outs;

    NeuronType(IdType i, unsigned ln, NeuronWeightType w)
      : Id(i), LayerNum(ln), W(w),
        dW(NeuronWeightType(0)), EvalSigCalled(false)
    { }
    NeuronType(IdType i, unsigned ln, DendronType* in,
               DendronType* out, NeuronWeightType w)
      : Id(i), LayerNum(ln), W(w),
        dW(NeuronWeightType(0)), EvalSigCalled(false) {
      Ins.push_back(in);
      Outs.push_back(out);
    }

    bool operator==(const NeuronType& n) const {
      return (n.LayerNum == LayerNum) && (n.Ins == Ins)
              && (n.Outs == Outs) && NeuronDist.CloseEnough(n.W, W);
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

    // For neurons weights are calculated 'not' assigned.
    // When it is assigned that means the weight will not be calculated
    // and evalOp will be invoked directly.
    void SetWeight(NeuronWeightType w) {
      assert(Valid(w));
      EvalSigCalled = true;
      W = w;
    }

    // Removes the entry of Dendron \d from the connections.
    void Disconnect(DendronType* d, Direction dir) {
      assert(dir != Direction::InvalidDirection);
      // Assume that incoming dendron is to be removed.
      DendronsRefVecType* ToBe = &Ins;
      if (dir == Direction::Outwards)
        ToBe = &Outs;
      DendronsRefVecType::iterator it = std::find(ToBe->begin(), ToBe->end(), d);
      assert(it != ToBe->end());
      ToBe->erase(it);
    }
    void Disconnect(DendronType* d) {
      Disconnect(d, Direction::Inwards);
      Disconnect(d, Direction::Outwards);
    }

    // This Neuron ---- n
    bool IsConnectedForward(NeuronType* n) {
      for (auto it = Outs.begin(); it != Outs.end(); ++it)
        if ((*it)->Out == n)
          return true;
      return false;
    }

    // This Neuron ---- n
    bool IsConnectedBackward(NeuronType* n) {
      for (auto it = Ins.begin(); it != Ins.end(); ++it)
        if ((*it)->In == n)
          return true;
      return false;
    }

    Direction IsConnected(NeuronType* n) {
      if (IsConnectedForward(n))
        return Direction::Outwards;
      if (IsConnectedBackward(n))
        return Direction::Inwards;
      return Direction::InvalidDirection;
    }

    Direction IsConnected(DendronType* d) {
      DendronsRefVecType::iterator it = std::find(Ins.begin(), Ins.end(), d);
      if (it != Ins.end())
        return Direction::Inwards;
      it = std::find(Outs.begin(), Outs.end(), d);
      if (it != Outs.end())
        return Direction::Outwards;
      return Direction::InvalidDirection;
    }

    DendronWeightType EvalSig() {
      assert(!IsRootNeuron(Id) &&
          "Cannot use this function for root neurons");
      auto d_it = Ins.begin();
      NeuronWeightType Sum(0);
      while(d_it != Ins.end()) {
        DendronType* d = *d_it;
        // Sum(weight of dendron * input)
        Sum += (d->W)*(d->In->Output);
        ++d_it;
      }
      W = Sum;
      EvalSigCalled = true;
      return Sum;
    }

    // To be used by the inner neurons only
    // and only after EvalSig has been called on this neuron.
    DendronWeightType EvalOp(const Activation<NeuronWeightType>& act) {
      assert(!IsRootNeuron(Id) &&
          "Cannot use this function for root neurons");
      assert(EvalSigCalled && "First calculate the signal/ip on this neuron");
      Output = act.Act(W);
      DEBUG1(dbgs() << "\nEval n"<< Id
                    << "(W:" << W << ", Output:" << Output << ")");
      EvalSigCalled = false;
      return Output;
    }

    std::ostream& operator<<(std::ostream& os) {
      Print(os);
      return os;
    }
    template<typename Stream>
    void Print(Stream& s) {
      s << "\n\tn" << Id
        << " [ label = \"n" << Id << "(W:" << W << ")\"];";
    }
  };


  typedef std::list<NeuronType> NeuronsType;
  typedef std::list<DendronType> DendronsType;
  typedef std::list<NeuronType*> NeuronsRefListType;
  typedef std::vector<NeuronType*> NeuronsRefVecType;
  typedef std::set<NeuronType*> NeuronsRefSetType; 
  /**
   *  ,--In--Cen---  -----
   * N --In--Cen---  ----- Out
   *  `--In--Cen---  -----
   * N is the root-neuron (kind of placeholder) and all the pseudo-edges
   * coming out of N are Input Dendrons.
   * This makes it easy to evaluate the network uniformly.
   * The neurons are the Adders of the weight*input of input dendrons.
   * @note As of now, the bias neuron is the first input neuron.
   * @note level 1 neurons are meant only for forwarding the input.
   */
  class NeuralNetwork {
    NeuronsType Neurons;
    DendronsType Dendrons;
    /// @todo this when using ids to index neurons.
    typedef std::vector<std::vector<NeuronType*> > NeuronsByLayerType;
    typedef Activation<NeuronWeightType> ActivationFnType;

    const ActivationFnType* ActivationFn;
    NeuronsByLayerType NeuronsByLayer;
    // Root is always the first entry in Neurons.
    NeuronType* RootNeuron;
    NeuronType* OutNeuron;
    unsigned NumLayers;
    public:
      NeuralNetwork()
        : ActivationFn(nullptr), RootNeuron(nullptr), NumLayers(0)
      { }

      NeuralNetwork(const ActivationFnType* act)
        : ActivationFn(act), RootNeuron(nullptr), NumLayers(0)
      { }

      // Initializes the network with just one neuron.
      NeuralNetwork(NeuronType& n, const ActivationFnType* act)
        : ActivationFn(act) {
        assert(act);
        Neurons.push_back(n);
        RootNeuron = &*Neurons.begin();
        NumLayers = 0;
      }

      // Root (the placeholder) has unit weight
      // so that evaluating subsequent stages becomes regular.
      NeuronsType::iterator CreateRoot() {
        assert(RootNeuron == NULL);
        // Root is in the zeroth layer and has a weight 1.
        CreateNeuron(0, 1);
        RootNeuron = &*Neurons.begin();
        return Neurons.begin();
      }
      /*/ If the network size is known beforehand.
      void Resize(unsigned numNeurons, unsigned numDendrons) {
        Neurons.resize(numNeurons);
        Dendrons.resize(numDendrons);
      }*/

      void SetOutputNeuron(NeuronType& n) {
        OutNeuron = &n;
      }

      NeuronType& GetOutputNeuron() {
        assert(OutNeuron && "OutNeuron hasn't been initilized yet");
        return *OutNeuron;
      }

      // Root is always the first entry.
      NeuronsType::iterator GetRoot() {
        assert(RootNeuron);
        return Neurons.begin();
      }

      const ActivationFnType& GetActivationFunction() const {
        return *ActivationFn;
      }

      // Use the iterator, don't reuse it.
      // It might have been invalidated.
      /// @param ln = Layer number where this neuron will go.
      NeuronsType::iterator CreateNeuron(unsigned ln, NeuronWeightType w
                                         = NeuronWeightType(0)) {
        NeuronType n(GetNewId(), ln, w);
        if (NumLayers < ln)
          NumLayers = ln;
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
        assert(!i1->IsConnected(&*i2));
        // Any connection from root has to be a unit weight.
        if (&*i1 == RootNeuron)
          assert(w == DendronWeightType(1));
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

      NeuronWeightType GetOutput(const std::vector<DendronWeightType>& Ins) {
        assert(RootNeuron->Outs.size() == Ins.size());
        assert(ActivationFn && "Uninitialized activation function");
        auto ip = Ins.begin();
        NeuronsRefVecType NeuronRefs;
        // Root. First propagate the input multiplied by
        // input dendron weights to the layer 1 neurons.
        NeuronType* Current = RootNeuron;
        NeuronsRefSetType InNeuronRefs1;
        auto it = Current->Outs.begin();
        for (; it != Current->Outs.end(); ++it) {
          DendronType* d = *it;
          d->Out->SetWeight((*ip)*d->W);
          DEBUG1(dbgs() << "\nWt: " << d->W << ", ip:" << *ip;
                 d->Out->Print(dbgs()););
          d->Out->EvalOp(*ActivationFn);
          auto dit = d->Out->Outs.begin();
          // Only insert the layer2 neurons because
          // layer1 neurons are evaluated in this loop.
          for (; dit != d->Out->Outs.end(); ++dit) {
            InNeuronRefs1.insert((*dit)->Out);
          }
          ++ip;
        }
        /// @note: The output neuron might be evaluated multiple times.
        /// but that's okay for now because it keeps the algorithm simple.
        for(auto it = InNeuronRefs1.begin(); it != InNeuronRefs1.end(); ++it)
          NeuronRefs.push_back(*it);
        InNeuronRefs1.clear();
        NeuronWeightType op;
        while(!NeuronRefs.empty()) {
          // TODO: Optimize this. Get a pointer to the neuron being
          // inserted to and check for size() > 1 using the pointer
          // to the set. That way I can avoid a copy.
          DEBUG1(dbgs() << "\nPrinting the neurons inserted:";
            std::for_each(NeuronRefs.begin(), NeuronRefs.end(),
              [&](NeuronType* np){ np->Print(dbgs()); dbgs() << " "; });
          );

          std::for_each(NeuronRefs.begin(), NeuronRefs.end(),
              [&](NeuronType* N) {
                N->EvalSig();
                op = N->EvalOp(*ActivationFn);
                std::for_each(N->Outs.begin(), N->Outs.end(),
                     [&](DendronType* din){ InNeuronRefs1.insert(din->Out); });
            });
          NeuronRefs.clear();
          for(auto it = InNeuronRefs1.begin(); it != InNeuronRefs1.end(); ++it)
            NeuronRefs.push_back(*it);
          InNeuronRefs1.clear();
        }
        return op;
      }

      unsigned GetTotalLayers() const {
        return NumLayers;
      }

      const NeuronsByLayerType& GetNeuronsByLayer() const {
        return NeuronsByLayer;
      }

      void ClearNeuronsByLayer() {
        NeuronsByLayer.clear();
      }

      NeuronsByLayerType GenNeuronsByLayer() {
        assert(NeuronsByLayer.empty());
        NeuronsByLayer.resize(GetTotalLayers()+1);
        for (auto it = Neurons.begin(); it != Neurons.end(); ++it) {
          NeuronType& n = *it;
          NeuronsByLayer[n.LayerNum].push_back(&n);
          DEBUG0(dbgs() << "\nNeuron#" << n.Id << ", Layer" << n.LayerNum);
        }
        return NeuronsByLayer;
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
          << "[ label = \"d" << d.Id << "(" << d.W<< ")\"];";
      }

      /** Depth First traversal of network
        * @todo Insted of using a std::set use a vector/bit-vector
        * initialized to same number of elements as there are dendrons.
        * That will make find very fast and hence the printing.
        */
      template<typename Stream>
      void PrintConnectionsDFS(NeuronType& root, DendronsRefSetType& Printed,
                            Stream& s) {
        for (auto dp = root.Outs.begin(); dp != root.Outs.end();
             ++dp) {
          if (Printed.find(*dp) == Printed.end())
            PrintDendron(s, **dp);
          Printed.insert(*dp);
          // This should work now because self-feedback is not supported.
          PrintConnectionsDFS(*(*dp)->Out, Printed, s);
        }
      }
      // Use a BFS method to print the neural network.
      template<typename Stream>
      void PrintNNDigraph(NeuronType& root, Stream& s,
                           const std::string& Name= "") {
        DendronsRefSetType Printed;
        s << "\ndigraph " << Name <<  " {\n";
        PrintNeurons(s);
        PrintConnectionsDFS(root, Printed, s);
        s << "\n}\n";
      }
  };

  // Single Layer Feed Forward network.
  // ,----IN ----
  // N----IN ---- Out
  // `----IN ----
  // @param NumNeurons = Number of input layer neurons.
  NeuralNetwork CreateSLFFN(unsigned NumNeurons,
                            const Activation<NeuronWeightType>* act) {
    using namespace utilities;
    RNG rnd(-1, 1);
    NeuralNetwork nn(act);
    auto root = nn.CreateRoot();
    auto out = nn.CreateNeuron(2, 0);
    auto bias = nn.CreateNeuron(1, 0);
    nn.SetOutputNeuron(*out);
    nn.Connect(root, bias, 1.0);
    nn.Connect(bias, out, 1.0);
    for (unsigned i = 0; i < NumNeurons -1; ++i) {
      auto in = nn.CreateNeuron(1, 0);
      nn.Connect(root, in, 1.0);
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
  NeuralNetwork CreateTLFFN(unsigned NumNeurons,
                            const Activation<NeuronWeightType>* act) {
    using namespace utilities;
    RNG rnd(-1, 1);
    NeuralNetwork nn(act);
    std::vector<NeuronsType::iterator> L2Neurons;
    auto root = nn.CreateRoot();
    auto out = nn.CreateNeuron(3, 0);
    nn.SetOutputNeuron(*out);
    auto bias = nn.CreateNeuron(1, 0);
    // Any connection from root has to be a unit weight.
    nn.Connect(root, bias, 1.0);
    nn.Connect(bias, out, 1.0);
    // Connect bias neurons to each l2 neurons,
    // and each l2 neurons to out.
    for (unsigned i = 0; i < NumNeurons -1; ++i) {
      auto l2 = nn.CreateNeuron(2, 0);
      nn.Connect(bias, l2, 1.0);
      nn.Connect(l2, out, rnd.Get());
      L2Neurons.push_back(l2);
    }
    // For each layer1 neuron make connection with it
    // to all the layer 2 neurons.
    for (unsigned i = 0; i < NumNeurons -1; ++i) {
      auto l1 = nn.CreateNeuron(1, 0);
      nn.Connect(root, l1, 1.0);
      for (unsigned j = 0; j < NumNeurons -1; ++j) {
        nn.Connect(l1, L2Neurons[j], rnd.Get());
      }
    }
    return nn;
  }

  template<typename Inputs, typename Output>
  class ValidateOutput {
  private:
    NeuralNetwork& NN;
  public:
    ValidateOutput(NeuralNetwork& nn)
      :NN(nn)
    { }

    template<typename T>
    Output GetDesiredOutput(Evaluate<T>* eval, Inputs Ins) const {
      //return std::accumulate(Ins.begin(), Ins.end(), Evaluate<T>::init_value,
      //                       eval);
      typename Evaluate<T>::init_type init = eval->init_value;
      DEBUG2(dbgs() << "\tInit: " << init);
      std::for_each(Ins.begin(), Ins.end(),
                    [&init, &eval](typename Inputs::value_type val) {
                      init = eval->operator()(init, val);
                    });
      DEBUG2(dbgs() << ", Desired output: " << init);
      return init;
    }

    template<typename T>
    bool Validate(Evaluate<T>* eval, Inputs Ins, Output op) const {
       return GetDesiredOutput(eval, Ins) == op;
    }
  };
} // namespace ANN

#endif // ANN_NEURAL_NETWORK_H
