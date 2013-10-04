#ifndef ANN_NEURAL_NETWORK_H
#define ANN_NEURAL_NETWORK_H

#include<Debug.h>
#include<RandomGenerator.h>
#include<Activation.h>
#include<TrainingAlgorithms.h>
#include<TypeConversions.h>
#include<Distance.h>

#include<vector>
#include<list>
#include<set>
#include<algorithm>
#include<cassert>
#include<numeric>
#include<cmath>
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
    DendronsRefVecType Ins;
    DendronsRefVecType Outs;

    NeuronType(IdType i, unsigned ln, NeuronWeightType w)
      : Id(i), LayerNum(ln), W(w), dW(NeuronWeightType(0))
    { }
    NeuronType(IdType i, unsigned ln, DendronType* in,
               DendronType* out, NeuronWeightType w)
      : Id(i), LayerNum(ln), W(w), dW(NeuronWeightType(0)) {
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

    void SetWeight(NeuronWeightType w) {
      assert(Valid(w));
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

    // To be used by the inner neurons only.
    // Note that the neurons only store the weight but it returns
    // the activation to the caller.
    template <typename ActivationFnType>
    DendronWeightType EvalOp(const ActivationFnType& act) {
      assert(!IsRootNeuron(Id) &&
          "Cannot use this function for root neurons");
      auto d_it = Ins.begin();
      DendronWeightType Sum(0);
      while(d_it != Ins.end()) {
        DendronType* d = *d_it;
        // Sum(weight of dendron * input)
        Sum += (d->W)*(d->In->GetWeight());
        ++d_it;
      }
      W = Sum;
      Output = act.Act(Sum);
      return Output;
    }

    virtual NeuronWeightType GetWeight() const {
      return W;
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
  template<typename ActivationFnType>
  class NeuralNetwork {
    NeuronsType Neurons;
    DendronsType Dendrons;
    /// @todo this when using ids to index neurons.
    typedef std::vector<std::vector<NeuronType*> > NeuronsByLayerType;
    NeuronsByLayerType NeuronsByLayer;
    // Root is always the first entry in Neurons.
    NeuronType* RootNeuron;
    unsigned NumLayers;
    ActivationFnType ActivationFn;
    public:
      // Empty
      NeuralNetwork()
        : RootNeuron(NULL), NumLayers(0)
      { }

      // Initializes the network with just one neuron.
      NeuralNetwork(NeuronType& n) {
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

      // Root is always the first entry.
      NeuronsType::iterator GetRoot() {
        assert(RootNeuron);
        return Neurons.begin();
      }

      const ActivationFnType& GetActivationFunction() const {
        return ActivationFn;
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
        auto ip = Ins.begin();
        NeuronsRefSetType NeuronRefs;
        // Root. First propagate the input multiplied by
        // input dendron weights to the layer 1 neurons.
        NeuronType* Current = RootNeuron;
        std::for_each(Current->Outs.begin(), Current->Outs.end(),
            [&NeuronRefs, &ip, this](DendronType* d) {
              d->Out->SetWeight((*ip)*d->W);
              DEBUG1(dbgs() << "\nWt: " << d->W << ", ip:" << *ip;
                d->Out->Print(dbgs()););
              std::for_each(d->Out->Outs.begin(), d->Out->Outs.end(),
                   [&](DendronType* din){ NeuronRefs.insert(din->Out); });
              ++ip;
            });
        NeuronsRefSetType InNeuronRefs1, InNeuronRefs2 = NeuronRefs;
        while(InNeuronRefs2.size() > 1) {
          // TODO: Optimize this. Get a pointer to the neuron being
          // inserted to and check for size() > 1 using the pointer
          // to the set. That way I can avoid a copy.
          InNeuronRefs1 = InNeuronRefs2;
          DEBUG1(dbgs() << "\nPrinting the neurons inserted:";
            std::for_each(InNeuronRefs2.begin(), InNeuronRefs2.end(),
              [&](NeuronType* np){ np->Print(dbgs()); dbgs() << " "; });
          );

          InNeuronRefs2.clear();
          std::for_each(InNeuronRefs1.begin(), InNeuronRefs1.end(),
              [&](NeuronType* N) {
                N->EvalOp(ActivationFn);
                std::for_each(N->Outs.begin(), N->Outs.end(),
                     [&](DendronType* din){ InNeuronRefs2.insert(din->Out); });
            });
        };
        assert(InNeuronRefs2.size() == 1);
        return (*InNeuronRefs2.begin())->EvalOp(ActivationFn);
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
        s << "digraph " << Name <<  " {\n";
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
  template<typename ActivationFnType>
  NeuralNetwork<ActivationFnType> CreateSLFFN(unsigned NumNeurons) {
    using namespace utilities;
    RNG rnd(-1, 1);
    NeuralNetwork<ActivationFnType> nn;
    auto root = nn.CreateRoot();
    auto out = nn.CreateNeuron(2, 0);
    for (unsigned i = 0; i < NumNeurons; ++i) {
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
  template<typename ActivationFnType>
  NeuralNetwork<ActivationFnType> CreateTLFFN(unsigned NumNeurons) {
    using namespace utilities;
    RNG rnd(-1, 1);
    NeuralNetwork<ActivationFnType> nn;
    std::vector<NeuronsType::iterator> L2Neurons;
    auto root = nn.CreateRoot();
    auto out = nn.CreateNeuron(3, 0);
    auto bias = nn.CreateNeuron(1, 0);
    // Any connection from root has to be a unit weight.
    nn.Connect(root, bias, 1.0);
    nn.Connect(bias, out, rnd.Get());
    // Connect bias neurons to each l2 neurons,
    // and each l2 neurons to out.
    for (unsigned i = 0; i < NumNeurons -1; ++i) {
      auto l2 = nn.CreateNeuron(2, 0);
      nn.Connect(bias, l2, rnd.Get());
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

  // Trains the neural network when the training algorithm
  // is provided. Each dendron is trained by the breadth first
  // traversal of the network starting from root node.
  // Assumptions: The dendrons have the weight and neurons
  // are the summers.
  template<typename NeuralNetworkType, typename TAType>
  class Trainer {
    NeuralNetworkType& NN;
    TAType TA;
    public:
    Trainer(NeuralNetworkType& nn)
      : NN(nn)
    { }
    const TAType& GetTrainingAlgorithm() {
      return TA;
    }
    const NeuralNetworkType& GetNeuralNetwork() const {
      return NN;
    }

    /// @todo: Put innovation number as well.
    // Training neuron means calculating the delta.
    void TrainOutputNeuron(NeuronType* n, NeuronWeightType desired_op) {
      TA.OutputNode(*n, desired_op, NN.GetActivationFunction());
    }
    void TrainHiddenNeuron(NeuronType* n) {
      TA.HiddenNode(*n, NN.GetActivationFunction());
    }
    void TrainDendron(DendronType *dp) {
      // weight update = alpha*input*delta
      //dp->W += dp->dW; // for momentum.
      dp->dW = -alpha*(dp->In->Output)*(dp->Out->dW);
      dp->W += dp->dW;
    }

    /** Breadth First traversal of network
      * @todo: Separate the training of level-1 neurons
      * in a separate loop. And then train all the subsequent
      * neurons. That would improve performance in
      * multi-layer networks with a lot of inner layers.
      */
    template<typename InputSet, typename OutputType>
    void TrainNetwork(const InputSet& Ins, OutputType desired_op) {
      NeuronType& root = *NN.GetRoot();
      assert(root.Outs.size() == Ins.size());
      NeuronsRefListType ToBeTrained;
      // level 1 neurons are meant only for forwarding the input.
      for (auto dp = root.Outs.begin(); dp != root.Outs.end(); ++dp) {
        ToBeTrained.push_back((*dp)->Out);
      }
      //ToBeTrained.push_back(&root);
      while (!ToBeTrained.empty()) {
        auto r = ToBeTrained.front();        
        for (auto dp = r->Outs.begin(); dp != r->Outs.end(); ++dp) {
          // incrementing the feed_ip works because input-size
          // is equal to the number of dendrons attached.
          //TrainOutputDendron(*dp, desired_op);
          ToBeTrained.push_back((*dp)->Out);
        }
        ToBeTrained.pop_front();
      }
    }
    // Only works for scalar outputs i.e. only one output neuron.
    template<typename InputSet, typename OutputType>
    void TrainNetworkBackProp(const InputSet& Ins, OutputType desired_op,
                              NeuronType& out) {
      NeuronType& root = *NN.GetRoot();
      assert(root.Outs.size() == Ins.size());
      assert(out.Outs.empty());
      NeuronsRefListType ToBeTrained;
      NeuronsRefSetType ToBeTrainedSet;
      TrainOutputNeuron(out, desired_op);
      for (auto dp = out.Ins.begin(); dp != out.Ins.end(); ++dp) {
        ToBeTrained.push_back((*dp)->In);
        TrainDendron(*dp);
      }
      // train all the neurons and dendrons until the input layer.
      // assuming the network is fully connected.
      for (auto np = ToBeTrained.begin(); np != ToBeTrained.end(); ++np) {
        TrainHiddenNeuron(*np);
        for (auto di = (*np)->Ins.begin(); di != (*np)->Ins.end(); ++di) {
          DendronType* dp = *di;
          DEBUG0(dbgs() << "\nTraining dendron:" << dp->Id);
          TrainDendron(dp);
          ToBeTrainedSet.insert(dp->In);
        }
      }
    }
  };

  // CRTP for evaluation functions.
  template<typename Evaluator, typename InitType>
  struct Evaluate {
    typedef InitType init_type;
    static const init_type init_value = Evaluator::init_value;
    template<typename T>
    T operator()(T t1, T t2) const {
      const Evaluator& e = static_cast<const Evaluator&>(*this);
      return e(t1, t2);
    }
  };

  struct BoolAnd : public Evaluate<BoolAnd, bool> {
    typedef bool init_type;
    static const init_type init_value = true;
    bool operator()(bool i1, bool i2) const {
      return i1 && i2;
    }
  };

  struct BoolOr : public Evaluate<BoolOr, bool> {
    typedef bool init_type;
    static const init_type init_value = false;
    bool operator()(bool i1, bool i2) const {
      return i1 || i2;
    }
  };

  struct BoolXor : public Evaluate<BoolXor, bool> {
    typedef bool init_type;
    static const init_type init_value = false;
    bool operator()(bool i1, bool i2) const {
      return i1 ^ i2;
    }
  };

  template<typename Evaluator, typename InitType=typename Evaluator::init_type>
  struct Evaluate;

  template<typename NeuralNetworkType, typename Inputs, typename Output>
  class ValidateOutput {
  private:
    NeuralNetworkType& NN;
  public:
    enum BooleanFunction {
      AND, // Init = true
      OR,  // Init = false
      XOR  // Init = false
    };
    ValidateOutput(NeuralNetworkType& nn)
      :NN(nn)
    { }

    template<typename T>
    Output GetDesiredOutput(Evaluate<T> eval, Inputs Ins) const {
      //return std::accumulate(Ins.begin(), Ins.end(), Evaluate<T>::init_value,
      //                       eval);
      typename Evaluate<T>::init_type init = Evaluate<T>::init_value;
      DEBUG2(dbgs() << "\tInit: " << init);
      std::for_each(Ins.begin(), Ins.end(),
                    [&init, &eval](typename Inputs::value_type val) {
                      init = eval(init, val);
                    });
      DEBUG2(dbgs() << ", Desired output: " << init);
      return init;
    }

    template<typename T>
    bool Validate(Evaluate<T> eval, Inputs Ins, Output op) const {
       return GetDesiredOutput(eval, Ins) == utilities::FloatToBool(op);
    }
  };
} // namespace ANN

#endif // ANN_NEURAL_NETWORK_H
