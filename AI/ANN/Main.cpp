#include "NetworkConfiguration.h"
#include "Debug.h"
#include <getopt.h>
#include <map>
#include <string>

void PrintCmdline() {
  std::cerr <<"\n\noptions: --activation Act --converge-method conv --cost-function cf"
            << "--network net --training-algo ta";
  std::cerr << "\n\n--activation [LinearAct|SigmoidAct]"
            << "\n--converge-method [GradientDescent|SimpleDelta]"
            << "\n--cost-function [BoolAnd|BoolOr|BoolXor]"
            << "\n--help"
            << "\n--network [SLFFN|TLFFN]"
            << "\n--training-algo [BackProp|FeedForward]"
            << "\n";
  std::cerr << "\nUsing GNU opt-parser, "
            << "either space or equals(=) works with long options.\n";
}

void PrintOptMap(OPT::OptmapType m) {
  auto& os = dbgs();
  std::for_each(m.begin(), m.end(),
                [&os](OPT::OptmapType::value_type v) {
                  std::cerr<< "\n" <<v.first << ":" << v.second;
                });
}

int main(int argc, char* argv[]) {
  using namespace OPT;
  int opt= 0;
  static struct option lopt[] = {
    {activation.c_str(),        required_argument, 0,  'a' },
    {converge_method.c_str(),   required_argument, 0,  'c' },
    {cost_function.c_str(),     required_argument, 0,  'C' },
    {help.c_str(),                    no_argument, 0,  'h' },
    {network.c_str(),           required_argument, 0,  'n' },
    {training_algo.c_str(),     required_argument, 0,  't' },
    {                    0,                     0, 0,   0  }
  };
  OptmapType optmap;
  int lidx =0;
  while ((opt = getopt_long_only(argc, argv,"",
                 lopt, &lidx )) != -1) {
    switch (opt) {
    case 'a':
      DEBUG2(dbgs() << "\nActivation Function:" << optarg);
      optmap[activation] = optarg;
      break;
    case 'c':
      DEBUG2(dbgs() << "\nConvergence Method:" << optarg);
      optmap[converge_method] = optarg;
      break;
    case 'C':
      DEBUG2(dbgs() << "\nCost Function:" << optarg);
      optmap[cost_function] = optarg;
      break;
    case 'h':
      PrintCmdline();
      return 0;
    case 'n':
      DEBUG2(dbgs() << "\nNetwork:" << optarg);
      optmap[network] = optarg;
      break;
    case 't':
      DEBUG2(dbgs() << "\nTraining Algorithm:" << optarg);
      optmap[training_algo] = optarg;
      break;
    default:
      std::cerr << "\nUnknown arguments. exiting...";
      PrintCmdline();
      return -1;
    }
  }

  PrintOptMap(optmap);

  ANN::NetworkConfiguration nc;
  if(nc.ValidateOptmap(optmap)) {
    nc.setup(optmap);
    while (!nc.VerifyTraining())
      nc.run();
  } else {
    DEBUG0(dbgs() << "\nInvalid/Insufficient options. exiting...");
    PrintCmdline();
    return -1;
  }
  return 0;
}
