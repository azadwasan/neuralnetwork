#ifndef CostFunctionMSE_H
#define CostFunctionMSE_H
#include "ICostFunction.h"
namespace EasyNN {
    class CostFunctionMSE : public ICostFunction
    {
        double evaluate(std::span<std::span<double>> x, std::span<double> y, const IHypothesis& hypothesis) override;
    };
}

#endif