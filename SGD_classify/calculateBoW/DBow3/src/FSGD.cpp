
#include "FSGD.h"

namespace DBoW3 {


	
	void FSGD::meanValue(const std::vector<FSGD::pDescriptor> &descriptors, 
			FSGD::TDescriptor &mean)
	{
		mean = cv::Mat::zeros(1, FSGD::L, CV_32F);
		float s = descriptors.size();

		std::vector<FSGD::pDescriptor>::const_iterator it;
		for(it = descriptors.begin(); it != descriptors.end(); ++it)
		{
			const FSGD::TDescriptor &desc = **it;
			float* p_mean = mean.ptr<float>(0);
			const float* p_desc = desc.ptr<float>(0);
			for(int i = 0; i < FSGD::L; i += 4)
			{
				p_mean[i  ] += p_desc[i  ] / s;
				p_mean[i+1] += p_desc[i+1] / s;
				p_mean[i+2] += p_desc[i+2] / s;
				p_mean[i+3] += p_desc[i+3] / s;
			}
		}
	}

	float FSGD::distance(const TDescriptor &a, const TDescriptor &b)
	{
		cv::Mat disMat = a - b;

		float dist = cv::norm(disMat);

		float d2 = dist * dist;

		return d2;
	}

	std::string FSGD::toString(const FSGD::TDescriptor &a)
	{
		std::stringstream ss;
		for(int i = 0; i < FSGD::L; ++i)
		{
			ss << a.at<float>(i) << " ";
		}
		return ss.str();
	}

	void FSGD::fromString(FSGD::TDescriptor &a, const std::string &s)
	{
		a = cv::Mat::zeros(1, FSGD::L, CV_32F);

		std::stringstream ss(s);
		for(int i = 0; i < FSGD::L; ++i)
		{
			ss >> a.at<float>(i);
		}
	}

	void FSGD::toMat32F(const std::vector<TDescriptor> &descriptors, 
			cv::Mat &mat)
	{
		if(descriptors.empty())
		{
			mat.release();
			return;
		}

		const int N = descriptors.size();
		const int L = FSGD::L;

		mat.create(N, L, CV_32F);

		for(int i = 0; i < N; ++i)
		{
			const TDescriptor& desc = descriptors[i];
			float *p = mat.ptr<float>(i);
			const float* p_desc = desc.ptr<float>(0);
			for(int j = 0; j < L; ++j, ++p)
			{
				*p = p_desc[j];
			}
		} 
	}

} // namespace DBoW3
