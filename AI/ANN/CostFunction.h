#ifndef ANN_COST_FUNCTION_H
#define ANN_COST_FUNCTION_H

namespace ANN {
  template<typename T>
  struct Evaluate {
    typedef T init_type;
    const init_type init_value;
    Evaluate(init_type i)
      : init_value(i)
    { }
    virtual T operator()(T t1, T t2) const = 0;
  };

  struct BoolAnd : public Evaluate<bool> {
    BoolAnd()
      : Evaluate(true)
    { }
    bool operator()(bool i1, bool i2) const {
      return i1 && i2;
    }
  };

  struct BoolOr : public Evaluate<bool> {
    BoolOr()
      : Evaluate(false)
    { }
    bool operator()(bool i1, bool i2) const {
      return i1 || i2;
    }
  };

  struct BoolXor : public Evaluate<bool> {
    BoolXor()
      : Evaluate(false)
    { }
    bool operator()(bool i1, bool i2) const {
      return i1 ^ i2;
    }
  };
} // namespace ANN

#endif // ANN_COST_FUNCTION_H
