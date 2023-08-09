// EasyNNConsole.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "LinearHypothesis.h"
#include "PythonPlugin.h"
#include "DataChannel.h"
#include "Algorithms.h"

int main()
{
    EasyNN::LinearHypothesis LH{};
    std::vector<double> values{ 1, 2, 3, 4, 5 };
    //LH.evaluate(values);
    std::cout << "Hello World!\n";
    try {
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        EasyNNPyPlugin::DataChannel::getRegressionData(X, y, 5, 2, 0.1);
        auto expectedParameters = EasyNNPyPlugin::Algorithms::RunGD(X, y, 3);
    }
    catch (std::exception e) {
        std::cout << "Python script execution failed!" << e.what() << std::endl;
    }
}
