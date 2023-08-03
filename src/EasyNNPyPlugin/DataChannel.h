#ifndef DataChannel_H
#define DataChannel_H
#include <vector>

namespace EasyNNPyPlugin {
	class DataChannel
	{
	public:
		static void getRegressionData(std::vector<std::vector<double>>& X, std::vector<double>& y, size_t nSamples, size_t nFeatures, double noise);
	};
}

#endif