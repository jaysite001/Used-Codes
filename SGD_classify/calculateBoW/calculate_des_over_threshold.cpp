#include "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle_localization/common/Thirdparty/DBow3/src/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>

using namespace cv;
using namespace std;

/***************************************************
 * 本节演示了如何根据前面训练的字典计算相似性评分
 * ************************************************/
int main(int argc, char **argv)
{
    // 读取字典
    cout << "reading database" << endl;
    DBoW3::Vocabulary vocab("/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/vocabulary.yml.gz");
    if (vocab.empty())
    {
        cerr << "Vocabulary does not exist." << endl;
        return 1;
    }

    //获取出现率超过阈值的Word
    string idx_over_threshold_path = "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/idx_over_threshold.txt";
    std::ifstream idx_over_threshold_file(idx_over_threshold_path);
    string s_idx_over_threshold;
    int idx_over_threshold;
    vector<int> WordID_over_threshold;
    while (getline(idx_over_threshold_file, s_idx_over_threshold))
    {
        idx_over_threshold = atoi(s_idx_over_threshold.c_str());
        WordID_over_threshold.push_back(idx_over_threshold);
    }

    // 輸入要匹配的描述符
    string desfilename = "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/SavedDesBA.txt";
    std::ifstream desfile(desfilename.c_str(), ios_base::in);
    istream_iterator<float> begin(desfile);
    istream_iterator<float> end;
    vector<float> inData(begin, end);
    desfile.close();
    cv::Mat des1 = cv::Mat(inData, true);
    cv::Mat descriptors = des1.reshape(0, inData.size() / 48);

    //计算每个描述符对应的 WORD
    std::string label_file_path = "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/label_over_threshold.txt";
    std::ofstream label_file(label_file_path);

    for (int r = 0; r < descriptors.rows; r++)
    {
        DBoW3::WordId id;
        DBoW3::WordValue w;
        vocab.transform(descriptors.row(r), id, w);

        vector<int>::iterator it = find(WordID_over_threshold.begin(), WordID_over_threshold.end(), id);
        if (it != WordID_over_threshold.end())
        {
            //vec中存在value值
            label_file << 1 << "\n";
        }
        else
        {
            //vec中不存在value值
            label_file << 0 << "\n";
        }
    }
    return 0;
}