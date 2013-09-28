#ifndef ANN_TRAINING_ALGORITHMS_H
#define ANN_TRAINING_ALGORITHMS_H

#include "RandomGenerator.h"

namespace ANN {
  // Keeping alpha low prevents oscillation.
  const float alpha = 0.02;
  utilities::RNG noise(0, alpha);
  template<typename AlgorithmType>
  struct TrainingAlgorithm {
    template<typename T>
    T operator()(T weight, T input, T desired_op) const {
      const AlgorithmType& a = static_cast<const AlgorithmType&>(*this);
      return a(weight, input, desired_op);
    }
  };

  struct GradientDescent : public TrainingAlgorithm<GradientDescent> {
    template<typename T>
    T operator()(T weight, T input, T desired_op) const {
      return weight + alpha*weight*(input - desired_op + noise.Get());
    }
  };
} // namespace ANN
#endif // ANN_TRAINING_ALGORITHMS_H
