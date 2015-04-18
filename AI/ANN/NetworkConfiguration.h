#ifndef NETWORKCONFIGURATION_H
#define NETWORKCONFIGURATION_H

#include "TrainingAlgorithms.h"

#include <map>
#include <string>

namespace OPT {
  typedef std::map<std::string, std::string> OptmapType;
  std::string activation = "activation";
  std::string converge_method = "converge-method";
  std::string cost_function = "cost-function";
  std::string help = "help";
  std::string network = "network";
  std::string training_algo = "training-algo";
} // namespace OPT

namespace ANN {

  class NetworkConfiguration {
    typedef ValidateOutput<std::vector<bool>, bool> ValidatorType;
    typedef std::map<std::string, Activation<NeuronWeightType>* >
            ActivationFunctionsType;
    typedef std::map<std::string, ConvergenceMethod*> ConvergenceMethodsType;
    typedef std::map<std::string, Evaluate<bool>* > EvaluatorsType;
    Evaluate<bool>* CostFunction;

    bool validated;
    Trainer* T;
    ValidatorType* Validator;
    float alpha;
    const int ip_size;
    unsigned times_trained;
    ActivationFunctionsType ActivationFunctions;
    //std::map<std::string, TrainingAlgorithm> TrainingAlgorithms;
    ConvergenceMethodsType ConvergenceMethods;
    EvaluatorsType Evaluators;
    NeuralNetwork NN;

  public:
    typedef std::map<std::string, std::string> OptMapType;

    NetworkConfiguration()
      : validated(false), T(nullptr), Validator(nullptr), alpha(0.01),
        ip_size(3), times_trained(0) {

      /// @todo Rather than generating by default,
      /// make the construction on-demand.
      ActivationFunctions["LinearAct"] = new LinearAct<NeuronWeightType>;
      ActivationFunctions["SigmoidAct"] = new SigmoidAct<NeuronWeightType>;
      ConvergenceMethods["SimpleDelta"] = new SimpleDelta;
      ConvergenceMethods["GradientDescent"] = new GradientDescent;
      Evaluators["BoolAnd"] = new BoolAnd;
      Evaluators["BoolOr"] = new BoolOr;
      Evaluators["BoolXor"] = new BoolXor;
    }

    ~NetworkConfiguration() {
      if (T)
        delete T;
      if (Validator)
        delete Validator;
      std::for_each(ActivationFunctions.begin(), ActivationFunctions.end(),
                    [](ActivationFunctionsType::value_type v){
                      delete v.second;
                    });
      std::for_each(ConvergenceMethods.begin(), ConvergenceMethods.end(),
                    [](ConvergenceMethodsType::value_type v){
                      delete v.second;
                    });
    }

    virtual void setup(OptMapType& optmap) {
      // Create a single layer feed forward neural network.
      NN = CreateTLFFN(ip_size, ActivationFunctions[optmap[OPT::activation]]);
      NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
      // Choose the training algorithm.
      ConvergenceMethod* CM = ConvergenceMethods[optmap[OPT::converge_method]];
      assert(CM);
      T = new Trainer(NN, CM, alpha);
      // Validation of the output.
      Validator = new ValidatorType(NN);
      CostFunction = Evaluators[optmap[OPT::cost_function]];
      assert(CostFunction);
    }

    virtual void run() {
      using namespace utilities;
      T->SetAlpha(alpha);
      DEBUG0(dbgs() << "\nTraining with alpha:" << alpha);
      for (unsigned i = 0; i < 10;) {
        std::vector<bool> RS = GetRandomizedSet(BooleanSampleSpace, ip_size-1);
        std::vector<float> RSF = BoolsToFloats(RS);
        // The last input is the bias.
        RSF.insert(RSF.begin(), -1);
        DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RSF));
        //NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
        auto op = NN.GetOutput(RSF);
        auto bool_op = FloatToBool(op);
        auto desired_op = Validator->GetDesiredOutput(CostFunction, RS);
        // Is the output same as desired output?
        if (!Validator->Validate(CostFunction, RS, bool_op)) {
          DEBUG0(dbgs() << "\nLearning (" << op << ", "
                        << bool_op << ", "
                        << desired_op << ")");
          //NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
          // No => Train
          T->TrainNetworkBackProp(RSF, desired_op);
          ++times_trained;
          i = 0;
          //NN.PrintNNDigraph(*NN.GetRoot(), std::cout);
        } else {
          ++i; // Increment trained counter.
          DEBUG0(dbgs() << "\tTrained (" << op << ", " << bool_op << ")");
        }
      }
    }

    virtual bool VerifyTraining() {
      using namespace utilities;
      bool trained = true;
      DEBUG0(dbgs() << "\nPrinting after training");
      for (unsigned i = 0; i < 20; ++i) {
        std::vector<bool> RS = GetRandomizedSet(BooleanSampleSpace, ip_size-1);
        std::vector<float> RSF = BoolsToFloats(RS);
        // The last input is the bias.
        RSF.insert(RSF.begin(), -1);
        auto op = NN.GetOutput(RSF);
        DEBUG0(dbgs() << "\nSample Inputs:"; PrintElements(dbgs(), RSF));
        if (Validator->Validate(CostFunction, RS, FloatToBool(op)))
         DEBUG0(dbgs() << "\tTrained (" << op << ", " << FloatToBool(op) << ")");
        else {
          // double the training rate.
          alpha = alpha < 0.4 ? 2*alpha : alpha;
          trained = false;
          DEBUG0(dbgs() << "\tUnTrained: " << op);
          break;
        }
      }
      DEBUG0(dbgs() << "\nTrained for " << times_trained << " cycles.");
      return trained;
    }

    bool ValidateOptmap(OptMapType& optmap) {
      using namespace OPT;
      if (optmap[activation] != "LinearAct" ||
          optmap[activation] != "SigmoidAct")
        return false;
      if (optmap[converge_method] != "GradientDescent" ||
          optmap[converge_method] != "SimpleDelta")
        return false;
      if (optmap[cost_function] != "BoolAnd" ||
          optmap[cost_function] != "BoolOr" ||
          optmap[cost_function] != "BoolXor")
        return false;
      if (optmap[network] != "SLFFN" ||
          optmap[network] != "TLFFN")
        return false;
      if (optmap[training_algo] != "BackProp" ||
          optmap[training_algo] != "FeedForward")
        return false;
      validated = true;
      return true;
    }
  };
}
#endif // NETWORKCONFIGURATION_H
