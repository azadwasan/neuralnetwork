#ifndef DataChannel_H
#define DataChannel_H
#include <vector>
#include "Common.h"

namespace EasyNNPyPlugin {
	class DataChannel
	{
	public:
		static void getRegressionData(std::vector<std::vector<double>>& X, std::vector<double>& y, size_t nSamples, size_t nFeatures, double noise);
		static void getClassificationData(std::vector<std::vector<double>>& X, std::vector<double>& y, size_t nSamples = 100, size_t nFeatures = 2, size_t redundantFeatures = 0, size_t clustersPerClass = 1, size_t randomState = 42);
	};
}

#endif