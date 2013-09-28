#ifndef UTILITIES_DISTANCE_H
#define UTILITIES_DISTANCE_H

#include<cmath>

namespace utilities {
static constexpr float FPDelta = 0.000001;

template<typename DT>
struct Distance {
  template<typename T>
  T operator()(T t1, T t2) const {
    const DT& dt = static_cast<const DT&>(*this);
    return dt(t1, t2);
  }

  template<typename T>
  bool CloseEnough(T t1, T t2) const {
    const DT& dt = static_cast<const DT&>(*this);
    return dt.CloseEnough(t1, t2);
  }
};

struct IntegralDistance : public Distance<IntegralDistance> {
  int operator()(int t1, int t2) const {
    return t1 - t2;
  }

  bool CloseEnough(int t1, int t2) const {
    return t1 == t2;
  }
};

struct FloatingPointDistance : public Distance<IntegralDistance> {
  float operator()(float t1, float t2) const {
    return t1 - t2;
  }

  bool CloseEnough(float t1, float t2) const {
    return std::fabs(t1 -t2) < FPDelta;
  }
};

} // namespace utilities
#endif // UTILITIES_DISTANCE_H
