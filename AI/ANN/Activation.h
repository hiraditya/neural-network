#ifndef ANN_ACTIVATION_FUNCTION_H
#define ANN_ACTIVATION_FUNCTION_H
#include<Debug.h>
#include<cmath>

namespace ANN {
  // Activation functions designed in the form of CRTP.
  template<typename T>
  class Activation {
    public:
      // @TODO: Try this: T operator()(T t) {
      // FAQ: Even if I do not write this function as
      // template of WeightType and put T instead of WeightType, it works.
      // I don't quite understand why?
      //
      template<typename WeightType>
      WeightType Act(WeightType t) const {
        const T& ref = static_cast<const T&>(*this);
        return ref.Act(t);
      }
  };

  template<typename WeightType>
  class LinearAct : public Activation<LinearAct<WeightType> > {
    public:
      WeightType Act(WeightType w) const {
        return w;
      }
  };

  template<typename WeightType>
  class SigmoidAct : public Activation<SigmoidAct<WeightType> > {
    public:
      WeightType Act(WeightType w) const {
        DEBUG2(dbgs() << "\nSigmoid function Input: " << w
                      << ", Output:" << std::tanh(w));
        return std::tanh(w);
      }
  };
} // namespace ANN



#endif // ANN_ACTIVATION_FUNCTION_H

