#ifndef UTILITIES_RNG_H
#define UTILITIES_RNG_H
#include<random>
#include<cmath>
#include<vector>

namespace utilities {
  typedef float RNType;

  std::vector<bool> BooleanSampleSpace{0, 1};

  class RNG {
    double Min, Max;
    std::random_device rd;
    std::mt19937 e2;
    std::uniform_real_distribution<> dist;

    public:
    RNG(double min, double max)
      : Min(min), Max(max), e2(rd()), dist(min, max)
    {}
    RNType Get() {
      return dist(e2);
    }

    unsigned GetLowerBound() {
      return std::floor(dist(e2));
    }

    unsigned GetUpperBound() {
      return std::ceil(dist(e2));
    }

    RNType GetBoolean() {
      return dist(e2) >= (Max-Min)/2.0 ? 1 : 0;
    }
  };

  // Get a vector of randomly selected Size elements from sample space.
  template<typename T>
  std::vector<T> GetRandomizedSet(const std::vector<T>& SampleSpace,
                                  unsigned Size) {
    RNG rng(0, SampleSpace.size());
    std::vector<T> RandomizedSet;
    for (size_t sz = 0; sz < Size; ++sz) {
      RandomizedSet.push_back(SampleSpace[rng.GetBoolean()]);
    }
    return RandomizedSet;
  }
} // namespace utilities

#endif // UTILITIES_RNG_H
