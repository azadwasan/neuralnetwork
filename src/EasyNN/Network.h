#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"

namespace EasyNN {
	class Network
	{
	public:
		void runNetwork();

		void setInput(Layer&& layer);
		void setInput(const Layer& layer);
		void setInput(std::vector<double>&& values);
		void setInput(const std::vector<double>& values);

		void addInputLayer(std::vector<double>&& values, std::shared_ptr<ICostFunction> costFunc = nullptr);
		void addInputLayer(const std::vector<double>& values, std::shared_ptr<ICostFunction> costFunc = nullptr);

		void addLayer(Layer&& layer) { layers.emplace_back(std::move(layer)); }
		void addLayer(const Layer& layer) { layers.emplace_back(layer); }
		void addLayer(std::vector<double>&& values, std::shared_ptr<ICostFunction> costFunc = nullptr);
		void addLayer(const std::vector<double>& values, std::shared_ptr<ICostFunction> costFunc = nullptr);

		const Layer& getOutputLayer();
		const std::vector<double>& getOutputValues();
	private:
		std::vector<Layer> layers;
	};
}


#endif