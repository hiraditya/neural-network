#ifndef ANN_TRAINING_ALGORITHMS_H
#define ANN_TRAINING_ALGORITHMS_H

#include "RandomGenerator.h"
#include "Activation.h"
#include <cassert>

namespace ANN {
  // Training algorithms only calculate the deltas.
  template<typename AlgorithmType>
  struct TrainingAlgorithm {
    template<typename DendronType, typename NeuronWeightType,
             typename ActivationFnType>
    void OutputNode(DendronType& d, NeuronWeightType desired_op,
                    const ActivationFnType& act) const {
      const AlgorithmType& a = static_cast<const AlgorithmType&>(*this);
      a.OutputNode(d, desired_op, act);
    }
    template<typename NeuronType, typename NeuronWeightType,
             typename ActivationFnType>
    void HiddenNode(NeuronType& n,
                    const ActivationFnType& act) const {
      const AlgorithmType& a = static_cast<const AlgorithmType&>(*this);
      a.HiddenNode(n, act);
    }
  };

  struct GradientDescent : public TrainingAlgorithm<GradientDescent> {
    template<typename NeuronType, typename NeuronWeightType,
             typename ActivationFnType>
    void OutputNode(NeuronType& n, NeuronWeightType desired_op,
                    const ActivationFnType& act) const {
      auto error = (n.Output - desired_op);
      // mean square error
      /// @todo Optimize when activation is tanh, in that case
      /// n.dW = 2*error*(1-n.Output*n.Output);
      n.dW = 2*error*act.Deriv(n.W);
    }
    template<typename NeuronType, typename NeuronWeightType,
             typename ActivationFnType>
    void HiddenNode(NeuronType& n,  NeuronWeightType desired_op,
                    const ActivationFnType& act) const {
      // n.dW = deriv(output)*Sum(delta(output-neuron))
      n.dW = NeuronWeightType(0);
      for (auto it = n.Outs.begin(); it != n.Outs.end(); ++it) {
        n.dW += ((*it)->W)*((*it)->Out->dW);
      }
      /// @todo Optimize when activation is tanh, in that case
      /// n.dW = n.dW*(1-n.Output*n.Output);
      n.dW = (n.dW)*(act.Deriv(n.W));
    }
  };
  struct FeedForward : public TrainingAlgorithm<FeedForward> {
    template<typename NeuronType, typename NeuronWeightType,
             typename ActivationFnType>
    void OutputNode(NeuronType& n, NeuronWeightType desired_op,
                    const ActivationFnType& act) const {
      auto error = (n.Output - desired_op);
      // mean square error
      /// @todo Optimize when activation is tanh, in that case
      /// n.dW = 2*error*(1-n.Output*n.Output);
      n.dW = 2*error*act.Deriv(n.W);
    }
    template<typename NeuronType, typename NeuronWeightType,
             typename ActivationFnType>
    void HiddenNode(NeuronType& n,  NeuronWeightType desired_op,
                    const ActivationFnType& act) const {
      assert(0 && "No hidden node training for feed forward network");
    }
  };
} // namespace ANN
#endif // ANN_TRAINING_ALGORITHMS_H
