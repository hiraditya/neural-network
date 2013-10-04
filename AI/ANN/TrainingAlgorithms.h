#ifndef ANN_TRAINING_ALGORITHMS_H
#define ANN_TRAINING_ALGORITHMS_H

#include "RandomGenerator.h"
#include "Activation.h"

namespace ANN {
  // Keeping alpha low prevents oscillation.
  const float alpha = 0.01;
  utilities::RNG noise(-alpha/5.0, alpha/5.0);
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
      n.dW = 2*error*act.Deriv(n.Output);
    }
    template<typename NeuronType, typename NeuronWeightType,
             typename ActivationFnType>
    void HiddenNode(NeuronType& n,
                    const ActivationFnType& act) const {
      n.dW(NeuronWeightType(0));
      std::for_each(n.Outs.begin(), n.Outs.end(),
                    [&n](typename NeuronType::DendronType_t* d) {
                      n.dW += d->W*d->Out->dW;
                    });
    /// @todo Optimize when activation is tanh, in that case
    /// n.dW = n.dW*(1-n.Output*n.Output);
    n.dW = n.dW*act.Deriv(n.W);
    }
  };
} // namespace ANN
#endif // ANN_TRAINING_ALGORITHMS_H
