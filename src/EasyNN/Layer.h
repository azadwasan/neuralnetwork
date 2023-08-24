#ifndef LAYER_H
#define LAYER_H

#include "CostFuntionLogistic.h"

#include <memory>
#include <vector>

namespace EasyNN {
	class Layer final
	{
	public:
		Layer(size_t neuronCount, size_t prevLayNeuronCount = 0, std::shared_ptr<ICostFunction> costFunc = nullptr)  :
			Layer{std::vector<double>(neuronCount), prevLayNeuronCount, costFunc}
		{
		}

		Layer(size_t neuronCount, const std::vector<std::vector<double>>& paramMatrix, std::shared_ptr<ICostFunction> costFunc = nullptr) :
			Layer{ std::vector<double>(neuronCount), parametersMatrix, costFunc }
		{
		}

		Layer(size_t neuronCount, std::vector<std::vector<double>>&& paramMatrix, std::shared_ptr<ICostFunction> costFunc = nullptr) :
			Layer{ std::vector<double>(neuronCount), std::move(parametersMatrix), costFunc }
		{
		}

		Layer(const std::vector<double>& neuronVals, size_t prevLayNeuronCount, std::shared_ptr<ICostFunction> costFunc) :
			neuronValues{neuronVals},
			parametersMatrix{
				prevLayNeuronCount > 0 ? // For input layers, we don't need a parametersMatrix.
				std::vector<std::vector<double>>(neuronValues.size(), std::vector<double>(prevLayNeuronCount)) :
				std::vector<std::vector<double>>()
			},
			costFunction{ costFunc == nullptr ? std::make_shared<CostFuntionLogistic>() : costFunc}
		{
		}

		Layer(const std::vector<double>& neuronVals, const std::vector<std::vector<double>>& paramMatrix, std::shared_ptr<ICostFunction> costFunc) :
			neuronValues{ neuronVals },
			parametersMatrix{paramMatrix},
			costFunction{ costFunc == nullptr ? std::make_shared<CostFuntionLogistic>() : costFunc }
		{
		}

		Layer(std::vector<double>&& neuronVals, size_t prevLayNeuronCount = 0, std::shared_ptr<ICostFunction> costFunc = nullptr) :
			neuronValues{ std::move(neuronVals) },
			parametersMatrix{
				prevLayNeuronCount > 0 ?
				std::vector<std::vector<double>>(neuronValues.size(), std::vector<double>(prevLayNeuronCount)) :
				std::vector<std::vector<double>>()
			},
			costFunction{ costFunc == nullptr ? std::make_shared<CostFuntionLogistic>() : costFunc }
		{
		}

		Layer(std::vector<double>&& neuronVals, std::vector<std::vector<double>>&& paramMatrix, std::shared_ptr<ICostFunction> costFunc = nullptr) :
			neuronValues{ std::move(neuronVals) },
			parametersMatrix{std::move(paramMatrix)},
			costFunction{ costFunc == nullptr ? std::make_shared<CostFuntionLogistic>() : costFunc }
		{
		}

		Layer(const Layer& other);
		Layer(Layer&& other) noexcept;
		Layer& operator=(const Layer& other) noexcept; 
		Layer& operator=(Layer&& other) noexcept;

		void computeOutput(const std::vector<double>& prevLayerNeurons) noexcept;

		void setNeuronValues(Layer&& values);
		void setNeuronValues(const Layer& values);
		void setNeuronValues(const std::vector<double>& values);
		void setNeuronValues(std::vector<double>&& values);
		const std::vector<double>& getNeuronValues() { return neuronValues; }
		size_t getNeuronCount() { return neuronValues.size(); }
		std::shared_ptr<ICostFunction> getCostFunction() { return costFunction; }
	private:
		std::vector<double> neuronValues;
		std::shared_ptr<ICostFunction> costFunction;
		std::vector<std::vector<double>> parametersMatrix{};
	};
}

#endif