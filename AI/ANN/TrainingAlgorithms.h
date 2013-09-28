#ifndef ANN_TRAINING_ALGORITHMS_H
#define ANN_TRAINING_ALGORITHMS_H

namespace ANN {
  float alpha = 0.02;
  template<typename AlgorithmType>
  class TrainingAlgorithm {
    public:
      template<typename T>
      T operator()(T weight, T input, T desired_op) const {
        const AlgorithmType& a = static_cast<const AlgorithmType&>(*this);
        return a(weight, input, desired_op);
      }
  };

  class GradientDescent : public TrainingAlgorithm<GradientDescent> {
    public:
    template<typename T>
    T operator()(T weight, T input, T desired_op) const {
        return weight + alpha*weight*(desired_op - input);
    }
  };
} // namespace ANN
#endif // ANN_TRAINING_ALGORITHMS_H
