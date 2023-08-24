#include "Network.h"

#include <stdexcept>>

using namespace EasyNN;

void Network::runNetwork()
{
	if (layers.size() > 1) {	// We need at least two leyers to run a network
		size_t index = 1;
		for (; index < layers.size(); index++) {
			layers[index].computeOutput(layers[index - 1].getNeuronValues());
		}
	}
}

void Network::setInput(Layer&& layer) {
	if (layers.size() > 0) {
		layers[0] = std::move(layer);
	}
	else {
		throw std::runtime_error("Input layer doens't exist!");
	}
}

void Network::setInput(const Layer& layer) {
	if (layers.size() > 0) {
		layers[0] = layer;
	}
	else {
		throw std::runtime_error("Input layer doens't exist!");
	}
}

void Network::setInput(std::vector<double>&& values){
	if (layers.size() > 0) {
		layers[0].setNeuronValues(values);
	}
	else {
		throw std::runtime_error("Input layer doens't exist!");
	}
}

void Network::setInput(const std::vector<double>& values) { 
	if (layers.size() > 0) {
		layers[0].setNeuronValues(values);
	}
	else {
		throw std::runtime_error("Input layer doens't exist!");
	}
}

void Network::addInputLayer(std::vector<double>&& values, std::shared_ptr<ICostFunction> costFunc /*= nullptr*/)
{
	if (layers.size() > 0) {
		throw std::runtime_error("Input layer already exists");
	}
	addLayer(std::move(values), costFunc);
}

void Network::addInputLayer(const std::vector<double>& values, std::shared_ptr<ICostFunction> costFunc /*= nullptr*/)
{
	if (layers.size() > 0) {
		throw std::runtime_error("Input layer already exists");
	}
	addLayer(values, costFunc);
}

void Network::addLayer(std::vector<double>&& values, std::shared_ptr<ICostFunction> costFunc /*= nullptr*/) {
	size_t prevLayerNeuronCount = 0;
	std::shared_ptr<ICostFunction> previousLayerCostFunction= nullptr;
	if (layers.size() > 0) {
		prevLayerNeuronCount		= layers[layers.size() - 1].getNeuronCount();
		previousLayerCostFunction	= layers[layers.size() - 1].getCostFunction();
	}
	layers.emplace_back(Layer{ std::move(values), prevLayerNeuronCount, previousLayerCostFunction });
}

void Network::addLayer(const std::vector<double>& values, std::shared_ptr<ICostFunction> costFunc /*= nullptr*/) {
	size_t prevLayerNeuronCount = 0;
	std::shared_ptr<ICostFunction> previousLayerCostFunction = nullptr;
	if (layers.size() > 0) {
		prevLayerNeuronCount		= layers[layers.size() - 1].getNeuronCount();
		previousLayerCostFunction	= layers[layers.size() - 1].getCostFunction();
	}
	layers.emplace_back(Layer{ std::move(values), prevLayerNeuronCount, previousLayerCostFunction });
}

const Layer& Network::getOutputLayer(){
	if (layers.size() > 0) {
		return layers[layers.size() - 1];
	}
	else {
		throw std::runtime_error("Network has zero layers!");
	}
}

const std::vector<double>& Network::getOutputValues() {
	if (layers.size() > 0) {
		return layers[layers.size() - 1].getNeuronValues();
	}
	else {
		throw std::runtime_error("Network has zero layers!");
	}
}
