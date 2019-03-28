#include "opencv2/opencv.hpp"
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
using namespace cv;

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


double euclidDist(Point first, Point second)
{
    double x = first.x - second.x;
    double y = first.y - second.y;
    double dist;

    dist = pow(x, 2) + pow(y, 2);
    dist = sqrt(dist);

    return dist;
}

double mapRange(int input,int input_start, int input_end, int output_start, int output_end) {

    double slope = 1.0 * (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);

}

int showImage = 1;

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


    Mat dest;
    Mat src_f;
    //original.convertTo(src_f,CV_32F);
    original.convertTo(src_f, CV_32F, 1.0/255, 0);

    std::vector<GaborParam> g_parameters;
/*
    g_parameters.push_back(makeGaborParam(10,18,59,77,98,21));
    g_parameters.push_back(makeGaborParam(10,18,57,97,98,21));
    g_parameters.push_back(makeGaborParam(10,18,57,96,96,7));
    g_parameters.push_back(makeGaborParam(10,6,40,176,226,8));
    g_parameters.push_back(makeGaborParam(10,6,42,95,221,8));
*/
    g_parameters.push_back(makeGaborParam(20, 8, 41, 0, 92, 4, 90));
    g_parameters.push_back(makeGaborParam(10, 6,40,2,93,2,90));
    g_parameters.push_back(makeGaborParam(10,10,36,75,66,58,120));
    g_parameters.push_back(makeGaborParam(10,7,25,43,107,15,24));
    g_parameters.push_back(makeGaborParam(20,11,50,104,80,58,158));



    for(std::vector<GaborParam>::iterator it = g_parameters.begin(); it != g_parameters.end(); ++it) {
        double sig = it->Sigma, th = theta(it->Theta), lm = lambda(it->Lambda), gm = it->Gamma, ps = psi(it->Psi);

        cv::Mat kernel = cv::getGaborKernel(cv::Size(it->kernel_size,it->kernel_size), sig, th, lm, gm, ps);
        cv::filter2D(src_f, dest, CV_32F, kernel);

        //cerr << dest(Rect(30,30,10,10)) << endl; // peek into the data

        Mat viz;
        dest.convertTo(viz,CV_8U,255.0/1.0);     // move to proper[0..255] range to show it
        //cerr << viz(Rect(30,30,10,10)) << endl; // peek into the data
        imshow("k",kernel);
        imshow("d",dest);
        imshow("v",viz);

        Mat resized(original.rows * 3, original.cols , CV_8UC1);

        original.copyTo( resized( Rect((0), (0/*y_offset*/), original.cols, original.rows) ) );
        viz.copyTo( resized( Rect((0), (original.rows/*y_offset*/), original.cols, original.rows) ) );

        cerr << viz(Rect(30,30,10,10)) << endl; // peek into the data

        Mat trash;
        threshold( viz, trash, it->Treshold, 255, 0);
        trash.copyTo( resized( Rect((0), (original.rows*2/*y_offset*/), original.cols, original.rows) ) );

        imshow("t",trash);
        imshow("Fin", resized);
        waitKey();
    }

    // FREE MEMORY
    src_f.release();
    dest.release();
    original.release();
    waitKey(0);

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

/*
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int pos_kernel_size=21;
int pos_sigma= 5;
int pos_lm = 50;
int pos_th = 0;
int pos_gamma= 0;
int pos_psi = 90;
int pos_trash = 90;

Mat src_f;
Mat dest;


bool exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

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


void Process(int , void *)
{

    int kernel_size=(pos_kernel_size-1)/2;
    cout << kernel_size;
    Size KernalSize(kernel_size,kernel_size);
    double Sigma = pos_sigma;
    double Lambda = 0.5+pos_lm/100.0;
    double Theta = pos_th*CV_PI/180;
    double psi = pos_psi*CV_PI/180;;
    double Gamma = pos_gamma;

    Mat kernel = getGaborKernel(KernalSize, Sigma, Theta, Lambda,Gamma,psi);
    filter2D(src_f, dest, CV_32F, kernel);
    imshow("Process window", dest);
    Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
    resize(kernel, Lkernel, Lkernel.size());
    Lkernel /= 2.;
    Lkernel += 0.5;
    imshow("Kernel", Lkernel);
    Mat mag;
    pow(dest, 2.0, mag);
    imshow("Mag", mag);

    Mat viz;
    dest.convertTo(viz,CV_8U,255.0/1.0);     // move to proper[0..255] range to show it
    //cerr << viz(Rect(30,30,10,10)) << endl; // peek into the data
    imshow("d",dest);
    imshow("v",viz);


    cerr << viz(Rect(30,30,10,10)) << endl; // peek into the data

    Mat trash;
    threshold( viz, trash, pos_trash, 255, 0);

    imshow("t",trash);
}

int main(int argc, char** argv)
{
    const string originalBase = "../../iris_NEW/";
    const string base = "../../iris_NEW_procesed/";
    parseCSV(originalBase);

    std::vector<std::vector<std::string> >::iterator row;
    std::vector<std::string>::iterator col;
    int cnt = 0;
    for (row = parsedCsv.begin(); row != parsedCsv.end(); row++, cnt++) {
        string file = base + row->at(0)+ "_f.jpg";
        if(!exists(file)) continue;

        Mat image = imread(file,0);
        cv::imshow("Src", image);

        image.convertTo(src_f, CV_32F, 1.0/255, 0);

        if (!pos_kernel_size%2)
        {
            pos_kernel_size+=1;
        }

        cv::namedWindow("Process window", 1);
        cv::createTrackbar("Sigma", "Process window", &pos_sigma, pos_kernel_size, Process);
        cv::createTrackbar("Lambda", "Process window", &pos_lm, 100, Process);
        cv::createTrackbar("Theta", "Process window", &pos_th, 180, Process);
        cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);
        cv::createTrackbar("Gamma", "Process window", &pos_gamma, 100, Process);
        cv::createTrackbar("Trashold", "Process window", &pos_trash, 255, Process);
        Process(0,0);
        waitKey(0);
    }

    return 0;
}
*/