#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"


namespace DBoW3 {

	class FSGD: protected FClass
	{
	public:
		static const int L = 64; 

		typedef cv::Mat TDescriptor; // CV_32F 
		typedef const TDescriptor *pDescriptor;

		static void meanValue(const std::vector<pDescriptor> &descriptors, 
				TDescriptor &mean);


		static float distance(const TDescriptor &a, const TDescriptor &b);

		static std::string toString(const FSGD::TDescriptor &a);
		static void fromString(FSGD::TDescriptor &a, const std::string &s);

		static void toMat32F(const std::vector<TDescriptor> &descriptors, 
			cv::Mat &mat);

	};

} // namespace DBoW3

