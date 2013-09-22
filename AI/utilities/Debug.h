#ifndef ANN_DEBUG_H
#define ANN_DEBUG_H

// This file sets up debugging infrastructure.
// This is mainly inspired from the LLVMs (www.llvm.org)
// method of debugging in the back end.
// There are various levels of debugging to facilitate
// what kind of debugging information user wants to print.
// For simplicity, there are four debugging modes defined.
// DEBUG0: Very high level of debugging. e.g., Interation between APIs
// DEBUG1:
// DEBUG2:
// DEBUG3: Lowest level of debugging. e.g., Printing contents of a container
// There is no strict requirement for what DEBUG a user can use,
// it is just to facilitate user see what is required while debugging.

#ifndef NDEBUG0
#define DEBUG0(ARG) do { ARG; } while(0)
#else
#define DEBUG0(ARG)
#endif

#ifndef NDEBUG1
#define DEBUG1(ARG) do { ARG; } while(0)
#else
#define DEBUG1(ARG)
#endif

#ifndef NDEBUG2
#define DEBUG2(ARG) do { ARG; } while(0)
#else
#define DEBUG2(ARG)
#endif

#ifndef NDEBUG3
#define DEBUG3(ARG) do { ARG; } while(0)
#else
#define DEBUG3(ARG)
#endif

#include<iostream>
#include<algorithm>

std::ostream& dbgs()
{
  return std::cout;
}

template<typename T>
void PrintElements(std::ostream& os, const T& t) {
  std::for_each(t.begin(), t.end(), [&os](typename T::value_type v){
      os << v << " ";
      });
  os << "\n";
}

template<typename T>
void PrintPointees(std::ostream& os, const T& t) {
  std::for_each(t.begin(), t.end(), [&os](typename T::value_type v){
      os << *v << " ";
      });
  os << "\n";
}

#endif // ANN_DEBUG_H
