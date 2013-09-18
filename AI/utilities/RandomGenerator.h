#ifndef UTILITIES_RNG_H
#define UTILITIES_RNG_H
#include<random>
#include<cmath>

namespace utilities {
  typedef float RNType;

  class RNG {
    std::random_device rd;
    std::mt19937 e2;
    std::uniform_real_distribution<> dist;

    public:
    RNG(unsigned min, unsigned max)
      : e2(rd()), dist(min, max)
    {}
    RNType Get() {
      return dist(e2);
    }
  };
} // namespace utilities

#endif // UTILITIES_RNG_H
