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

    // 輸入要匹配的描述符
    std::vector<cv::Mat> descriptors;
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
            descriptors.push_back(des2);
        }
        i++;
    }

    // we can compare the images directly or we can compare one image to a database
    // images :
    system("rm -r /media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Bowvectors");
    system("mkdir /media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Bowvectors");
    cout << "comparing images with images " << endl;
    for (int i = 0; i < descriptors.size(); i++)
    {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        string bowvecname = "/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Bowvectors/BowVec" + to_string(i) + ".txt";
        v1.saveM(bowvecname, vocab.size());
        std::cout << "Get BoW Vector: " << i << std::endl;
    }

    // or with database
    cout << "comparing images with database " << endl;
    DBoW3::Database db(vocab, false, 0);
    for (int i = 0; i < descriptors.size(); i++)
        db.add(descriptors[i]);
    cout << "database info: " << db << endl;
    for (int i = 0; i < descriptors.size(); i++)
    {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4); // max result=4
        cout << "searching for image " << i << " returns " << ret << endl
             << endl;
    }
    cout << "done." << endl;
    return 0;
}