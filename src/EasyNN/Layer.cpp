#include "Layer.h"
#include <stdexcept>

using namespace EasyNN;

Layer::Layer(const Layer& other) {
	neuronValues = other.neuronValues;
	costFunction = other.costFunction;
	parametersMatrix = other.parametersMatrix;
}

Layer::Layer(Layer&& other) noexcept {
	 neuronValues = std::move(other.neuronValues);
	 costFunction = other.costFunction;
	 parametersMatrix = std::move(other.parametersMatrix);
}

Layer& Layer::operator=(const Layer& other) noexcept{
	if (this != &other) {
		neuronValues = other.neuronValues;
		costFunction = other.costFunction;
		parametersMatrix = other.parametersMatrix;
	}
	return *this;
}

Layer& Layer::operator=(Layer&& other) noexcept {
	if (this != &other) {
		neuronValues = std::move(other.neuronValues);
		costFunction = other.costFunction;
		parametersMatrix = std::move(other.parametersMatrix);
	}
	return *this;
}


void Layer::computeOutput(const std::vector<double>& prevLayerNeurons) noexcept
{
	for (size_t index = 0; index < neuronValues.size(); index++) {
		neuronValues[index] = costFunction->getHypothesis().evaluate(prevLayerNeurons, parametersMatrix[index]);
	}
}

void Layer::setNeuronValues(Layer&& values)
{
	neuronValues = std::move(values.neuronValues);
}

void Layer::setNeuronValues(const Layer& values)
{
	neuronValues = values.neuronValues;
}

void Layer::setNeuronValues(const std::vector<double>& values)
{
	if (values.size() != neuronValues.size()) {
		throw std::invalid_argument("Passed neuron values vector size does not match layer neuron vectors size!");
	}
	neuronValues = values;
}

void EasyNN::Layer::setNeuronValues(std::vector<double>&& values)
{
	if (values.size() != neuronValues.size()) {
		throw std::invalid_argument("Passed neuron values vector size does not match layer neuron vectors size!");
	}
	neuronValues = std::move(values);
}
