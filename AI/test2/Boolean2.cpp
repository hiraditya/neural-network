#include<RandomGenerator.h>
#include<Debug.h>
#include<iostream>

// Test the randomization function.
int test1() {
  using namespace utilities;
  std::vector<float> SampleSpace{0, 1};
  std::vector<float> RS = GetRandomizedSet(SampleSpace, 2);
  DEBUG0(PrintElements(dbgs(), RS));
  return 0;
}
int test2() {
  using namespace utilities;
  std::vector<float> SampleSpace{0, 1, 2, 3};
  std::vector<float> RS = GetRandomizedSet(SampleSpace, 10);
  DEBUG0(PrintElements(dbgs(), RS));
  return 0;
}

int main() {
  test1();
  test2();
}
