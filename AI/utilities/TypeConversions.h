#ifndef UTILITIES_TYPECONVERSIONS_H
#define UTILITIES_TYPECONVERSIONS_H

#include"Distance.h"

#include<algorithm>
#include<vector>
#include<cmath>

namespace utilities {
  std::vector<float> BoolsToFloats(const std::vector<bool>& boolArray) {
    std::vector<float> C;
    std::for_each(boolArray.begin(), boolArray.end(),
                  [&C](bool b){ C.push_back(b); });
    return C;
  }
  bool FloatToBool(float f) {
    //FloatingPointDistance fp;
    return /*fp.CloseEnough(f, 0) || */ f > 0;
  }
} // namespace utilities

#endif // UTILITIES_TYPECONVERSIONS_H
