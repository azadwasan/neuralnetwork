// EasyNNConsole.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "LinearRegression.h"
#include "PythonPlugin.h"
#include "DataChannel.h"
#include "Algorithms.h"
#include "Plots.h"

int main()
{
    EasyNN::LinearRegression LH{};
    std::vector<double> values{ 1, 2, 3, 4, 5 };
    //LH.evaluate(values);
    std::cout << "Hello World!\n";
    try {
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        //EasyNNPyPlugin::DataChannel::getRegressionData(X, y, 5, 2, 0.1);
        //auto expectedParameters = EasyNNPyPlugin::Algorithms::RunGD(X, y, 3);
        EasyNNPyPlugin::DataChannel::getClassificationData(X, y);
        auto logisticRegressionFit = EasyNNPyPlugin::Algorithms::FitLogisticRegressionTF(X, y);
        EasyNNPyPlugin::Plots::PlotClassificationData(X, y, logisticRegressionFit);
    }
    catch (std::exception e) {
        std::cout << "Python script execution failed!" << e.what() << std::endl;
    }
}
