#include "opencv2/opencv.hpp"
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

std::vector<std::vector<std::string> > parsedCsv;

void parseCSV(const string base)
{

    std::ifstream  data(base + "iris_bounding_circles.csv");
    std::string line;

    int lin = 0;
    while(std::getline(data,line))
    {
        lin++;
        if(lin == 1) continue;
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        while(std::getline(lineStream,cell,','))
        {
            parsedRow.push_back(cell);
        }

        parsedCsv.push_back(parsedRow);
    }

    std::vector<std::vector<std::string> >::iterator row;
    std::vector<std::string>::iterator col;

};
bool exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}


int showImage = 0;

double lambda( int val) {
    return 0.5+val/100.0;
}
double theta( int val ) {
    return val*CV_PI/180;
}
double psi(int val) {
    return val*CV_PI/180;
}


struct GaborParam
{
    int kernel_size;
    int Sigma;
    int Lambda;
    int Theta;
    int Psi;
    int Gamma;
    int Treshold;
};

GaborParam makeGaborParam(int kernel_size,int Sigma,int Lambda,int Theta,int Psi,int Gamma, int Treshold)
{
    GaborParam ret;
    ret.kernel_size = kernel_size;
    ret.Sigma = Sigma;
    ret.Lambda = Lambda;
    ret.Theta = Theta;
    ret.Psi = Psi;
    ret.Gamma = Gamma;
    ret.Treshold = Treshold;
    return ret;
}

double flatten(string base,string saveBase, std::vector<std::string> data, int width, int height) {

    string file = base + data.at(0)+ "_f.jpg";
    if (!exists(file)) return -1;

    cout << data.at(0);

    Mat original = imread(file, IMREAD_GRAYSCALE);
    if(showImage) imshow("original", original);

    string saveFile = saveBase + data.at(0);
/*
    Ptr<SIFT> detector = SIFT::create();

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(original, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(original, keypoints, output);
    cv::imwrite("sift_result.jpg", output);
*/
    Mat dest;
    Mat src_f;

    original.convertTo(src_f, CV_32F, 1.0/255, 0);

    std::vector<GaborParam> g_parameters;

    g_parameters.push_back(makeGaborParam(20, 8, 41, 0, 92, 4, 90));
    for(int i = 0; i < 160; i+=10) {
        g_parameters.push_back(makeGaborParam(20, 8, 41, 0, i, 4, 90));
    }
/*
    g_parameters.push_back(makeGaborParam(10, 6,40,2,93,2,90));
    g_parameters.push_back(makeGaborParam(10,10,36,75,66,58,120));
    g_parameters.push_back(makeGaborParam(10,7,25,43,107,15,24));
    g_parameters.push_back(makeGaborParam(20,11,50,104,80,58,158));
*/

    cv::Mat finmat = cv::Mat::zeros(original.size(), CV_32F);
    cv::Mat conv = cv::Mat::zeros(original.size(), CV_32F);
    cv::Mat showis = cv::Mat::zeros(original.size(), CV_8U);
    int count = 0;
    for(std::vector<GaborParam>::iterator it = g_parameters.begin(); it != g_parameters.end(); ++it) {
        double sig = it->Sigma, th = theta(it->Theta), lm = lambda(it->Lambda), gm = it->Gamma, ps = psi(it->Psi);

        cv::Mat kernel = cv::getGaborKernel(cv::Size(it->kernel_size,it->kernel_size), sig, th, lm, gm, ps);
        cv::filter2D(src_f, dest, CV_32F, kernel);

        Mat viz;
        dest.convertTo(viz,CV_8U,255.0/1.0);     // move to proper[0..255] range to show it

        if(showImage)  imshow("k",kernel);
        if(showImage) imshow("d",dest);
        if(showImage) imshow("v",viz);

        Mat resized(original.rows * 3, original.cols , CV_8UC1);

        original.copyTo( resized( Rect((0), (0/*y_offset*/), original.cols, original.rows) ) );
        viz.copyTo( resized( Rect((0), (original.rows/*y_offset*/), original.cols, original.rows) ) );

       // cerr << viz(Rect(30,30,10,10)) << endl; // peek into the data

        Mat trash;
        threshold( viz, trash, it->Treshold, 255, 0);
        trash.copyTo( resized( Rect((0), (original.rows*2/*y_offset*/), original.cols, original.rows) ) );

        if(showImage) imshow("t",trash);
        if(showImage) imshow("Fin", resized);
        trash.convertTo(conv, CV_32F);
        finmat = ((conv + finmat));

        count++;
        if(count > 1) {
            finmat = finmat/2;
        }
    }
    finmat.convertTo(showis,CV_8U,255.0/1.0);
    if(showImage) imshow("Sum", showis);

    imwrite(saveFile, showis);

    if(showImage) waitKey();
    // FREE MEMORY
    src_f.release();
    dest.release();
    original.release();
    finmat.release();
    conv.release();
    showis.release();

    //waitKey(0);

}

int main( int argc, const char** argv )
{
    const string originalBase = "../../iris_NEW/";
    const string base = "../../iris_NEW_procesed/";
    const string saveBase = "../../iris_NEW_3/";
    parseCSV(originalBase);
    std::vector<std::vector<std::string> >::iterator row;
    std::vector<std::string>::iterator col;
    int cnt = 0;
    for (row = parsedCsv.begin(); row != parsedCsv.end(); row++, cnt++) {
        flatten(base, saveBase, *row, 365, 60);

        cout << "\n";
    }

    return 0;
}
