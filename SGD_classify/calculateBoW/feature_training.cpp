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
 * 训练字典
 * ************************************************/

int main(int argc, char **argv)
{
    bool useAllSGD = 0; //1 means use SGD after BA, 0 means use SGD from all frame
    std::vector<cv::Mat> descriptors;
    if (useAllSGD)
    {
        string filename = "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/SavedDesBA.txt";
        std::ifstream desfile(filename.c_str(), ios_base::in);
        if (!desfile.is_open())
        {
            cout << "读取文件失败" << endl;
            cv::waitKey(0);
            return 0;
        }
        istream_iterator<float> begin(desfile); //按 float 格式取文件数据流的起始指针
        istream_iterator<float> end;
        vector<float> inData(begin, end);
        desfile.close();
        cv::Mat des1 = cv::Mat(inData, true);
        cv::Mat des2 = des1.reshape(0, inData.size() / 48);
        descriptors.push_back(des2);
    }
    else
    {
        int i = 0;
        while (1)
        {
            string desfilename = "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Desfile/Des" + std::to_string(i) + ".txt";
            std::ifstream desfile(desfilename.c_str(), ios_base::in);
            if (!desfile.is_open())
            {
                break;
            }
            else
            {
                istream_iterator<float> begin(desfile);
                istream_iterator<float> end;
                vector<float> inData(begin, end);
                desfile.close();
                cv::Mat des1 = cv::Mat(inData, true);
                cv::Mat des2 = des1.reshape(0, inData.size() / 48);
                descriptors.push_back(des2.clone());
            }
            i++;
        }
    }

    // create vocabulary
    cout << "creating vocabulary ... " << endl;
    DBoW3::Vocabulary vocab(10, 5);
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    vocab.save("/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/vocabulary.yml.gz");
    vocab.save("/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/vocabulary.yml");
    cout << "done" << endl;
    cout << "vocab size:" << vocab.size() << endl;

    getchar();
    return 0;
}