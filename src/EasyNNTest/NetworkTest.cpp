#include "pch.h"
#include "CppUnitTest.h"

#include "Network.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EasyNNTest {
	TEST_CLASS(NetworkTest) {
public:
		void testRunner(EasyNN::Network& network, std::vector<double>& input, std::vector<EasyNN::Layer>& layers, size_t counter) {
			network.addInputLayer(input);
			for (auto& layer : layers) {
				network.addLayer(layer);
			}
			network.runNetwork();
			const auto& output = network.getOutputValues();


			auto msg = "Running Test # " + std::to_string(counter++) + ", Input = (";//
			for (auto in : input) {
				msg += std::to_string(in) + ", " ;
			}
			msg += "). Output = ";
			Logger::WriteMessage(msg.c_str());
			for (auto val : output) {
				Logger::WriteMessage(std::to_string(val).c_str());
			}
			Logger::WriteMessage("\n");
		}
		TEST_METHOD(AndFeedForwardNetwork) {
			std::vector<std::vector<double>> inputs { {0.0, 0.0}, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }};
			size_t counter = 1;
			for (auto& input : inputs) {
				EasyNN::Network network;
				EasyNN::Layer layer{std::vector<double>{0.0}, std::vector<std::vector<double>>{{-30.0, 20.0, 20.0}}};
				std::vector<EasyNN::Layer> layers{layer};
				testRunner(network, input, layers, counter++);
			}
			//network.addInputLayer({ 0.0, 0.0});
			//network.addLayer(std::move(layer));
			//network.runNetwork();
			//const auto& output = network.getOutputValues();
			//for (auto val : output) {
			//	Logger::WriteMessage(std::to_string(val).c_str());
			//}
		}
		TEST_METHOD(OrFeedForwardNetwork) {
			std::vector<std::vector<double>> inputs { {0.0, 0.0}, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }};
			size_t counter = 1;
			for (auto& input : inputs) {
				EasyNN::Network network;
				EasyNN::Layer layer{std::vector<double>{0.0}, std::vector<std::vector<double>>{{-10.0, 20.0, 20.0}}};
				std::vector<EasyNN::Layer> layers{layer};
				testRunner(network, input, layers, counter++);
			}
		}
		TEST_METHOD(NotFeedForwardNetwork) {
			std::vector<std::vector<double>> inputs { {0.0}, { 1.0}};
			size_t counter = 1;
			for (auto& input : inputs) {
				EasyNN::Network network;
				EasyNN::Layer layer{std::vector<double>{0.0}, std::vector<std::vector<double>>{{10.0, -20.0}}};
				std::vector<EasyNN::Layer> layers{layer};
				testRunner(network, input, layers, counter++);
			}
		}

		TEST_METHOD(XNORFeedForwardNetwork) {
			std::vector<std::vector<double>> inputs { {0.0, 0.0}, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }};
			size_t counter = 1;
			for (auto& input : inputs) {
				EasyNN::Network network;
				EasyNN::Layer layer{{0.0, 0.0}, {{-30.0, 20.0, 20}, {10, -20, -20}}};
				EasyNN::Layer layer2{std::vector<double>{0.0}, { {-10, 20, 20} } };
				std::vector<EasyNN::Layer> layers{layer, layer2};

				testRunner(network, input, layers, counter++);
			}
		}

	};
}