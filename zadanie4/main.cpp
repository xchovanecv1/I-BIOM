#include "opencv2/opencv.hpp"
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <algorithm>

#include <iostream>
#include <filesystem>


#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/*
 * https://picoledelimao.github.io/blog/2016/01/31/is-it-a-cat-or-dog-a-neural-network-application-in-opencv/
 * */

namespace fs = std::experimental::filesystem;

std::vector<std::vector<std::string> > parsedCsv;
typedef vector< tuple<string, string, int> > im_pair;

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

void flatImage( Mat& in) {
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            uint8_t val = in.at<uchar>(i, j);
            if (val > 128) {
                val = 255;
            } else {
                val = 0;
            }
            in.at<uchar>(i, j) = val;
        }
    }
}

void matFromOfset(Mat & in, Mat & out, int ofset) {
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            int j2 = (j + ofset) % in.cols;

            int val = in.at<uchar>(i, j);
            out.at<uchar>(i, j2) = val;

        }
    }
}

int hammingDist(Mat & im1, Mat & im1Mask, Mat & im2, Mat & im2Mask, int offset = 0) {
    int difs = 0;
    for (int i = 0; i < im1.rows; i++) {
        for (int j = 0; j < im1.cols; j++) {
            int j2 = (j + offset) % im1.cols;

            int val = im1.at<uchar>(i, j);
            int val2 = im2.at<uchar>(i, j2);
            int mask1 = im1Mask.at<uchar>(i, j);
            int mask2 = im2Mask.at<uchar>(i, j2);
            //cout << mask1 << " " << mask2 << endl;
            if(mask1 != 0 && mask2 != 0) {
                if(val != val2) {
                    difs++;
                }
            }

        }
    }
    return difs;
}

double flatten(string base,string mapBase, std::vector<std::string> data,std::vector<std::string> data2, int width, int height) {

    string file = base + data.at(0);
    string mapFile = mapBase + data.at(0)+ "_m.jpg";
    if (!exists(file)) return -1;
    if (!exists(mapFile)) return -1;

    string file2 = base + data2.at(0);
    string mapFile2 = mapBase + data2.at(0)+ "_m.jpg";
    if (!exists(file2)) return -1;
    if (!exists(mapFile2)) return -1;

    cout << data.at(0) << " " << data2.at(0) << endl;

    Mat original = imread(file, IMREAD_UNCHANGED);
    Mat mapOrig = imread(mapFile, IMREAD_UNCHANGED);
    Mat original2 = imread(file2, IMREAD_UNCHANGED);
    Mat mapOrig2 = imread(mapFile2, IMREAD_UNCHANGED);

    if(showImage) imshow("original", original);
    if(showImage) imshow("original map", mapOrig);
    if(showImage) imshow("original2", original2);
    if(showImage) imshow("original map2", mapOrig2);


    flatImage(original);
    flatImage(original2);
    flatImage(mapOrig);
    flatImage(mapOrig2);

    int min = INT_MAX;
    int min_ofset = 0;
    for(int ofs = 0; ofs < original.cols; ofs++) {
        int val = hammingDist(original, mapOrig, original2, mapOrig2, ofs);
        if(val < min) {
            min = val;
            min_ofset = ofs;
        }
    }
    cout << "Min dist: " << min << " Ofset" << min_ofset << endl;

    Mat fixed = cv::Mat::zeros(original.size(), CV_8U);
    Mat fixedMap = cv::Mat::zeros(original.size(), CV_8U);
    matFromOfset(original2, fixed, min_ofset);
    matFromOfset(mapOrig2, fixedMap, min_ofset);

    if(showImage) imshow("original2 FIXED", fixed);
    if(showImage) imshow("original2 MAP FIXED", fixedMap);
    /*
    for(int i=0; i<original.rows; i++)
        for(int j=0; j<original.cols; j++)
            std::cout << (int)original.at<uchar>(i,j) << std::endl;
*/
    if(showImage) waitKey();

    // FREE MEMORY

    original.release();

    //waitKey(0);

}

Mat sift_Extract(string base, std::vector<std::string> data, int maxDescs, int* label, char* lr) {

    string file = base + data.at(0) + "_f.jpg";
    string map = base + data.at(0) + "_m.jpg";
    Mat out;
    *lr = 0;
    *label = 0;

    if (exists(file) && exists(map)) {

        string name = data.at(0).substr(0, data.at(0).find("/"));
        cout << data.at(0) << endl;
        *label = stoi(name);
        *lr = data.at(0).substr(name.length()+1, 1)[0];

        cout << "class " << *label << "LR " << *lr << endl;

        Mat original = imread(file, IMREAD_UNCHANGED);
        Mat originalMap = imread(map, IMREAD_UNCHANGED);
        flatImage(originalMap);
        //if(showImage) imshow("original", original);
        //if(showImage) imshow("map", originalMap);
        Mat proces;
        original.copyTo(proces, originalMap);
        //if(showImage) imshow("proc", proces);

        Ptr<SIFT> detector = SIFT::create(maxDescs);

        Mat descriptors;

        std::vector<cv::KeyPoint> keypoints;
        detector->detect(proces, keypoints);
        detector->compute(original, keypoints, descriptors);

        if(descriptors.rows < maxDescs)  {
            *lr = 0;
            return out;
        }

        cout << descriptors.rows << " " <<descriptors.cols << endl;
        descriptors = descriptors(Rect(0,0,128,maxDescs));


        //cout << descriptors << endl;

        out = descriptors.reshape(1,1);

        cv::Mat output;
        cv::drawKeypoints(proces, keypoints, output);

        //imshow("Key", output);

        //waitKey(0);

        original.release();
        output.release();
    }
    return out;
}

void right_pairs(im_pair &in, string base, int max, bool incorrect = false) {

    char fr[100], sc[100];
    char side[1], other[1];

    int max_count = 0;

    for(int i= 1; i < 200; i++) {

        for(int g = 2; g <= 12; g++) {
            int g_idx = g;
            cout << g << endl;
            if (g <= 6) {
                side[0] = 'R';
                other[0] = 'L';
            } else {
                g_idx -= 5;
                side[0] = 'L';
                other[0] = 'R';
            }
            snprintf(fr, sizeof(fr), "%03d/%c/S1%03d%c%02d.jpg", i, side[0],i, side[0],1);
            if(incorrect) side[0] = other[0];
            snprintf(sc, sizeof(sc), "%03d/%c/S1%03d%c%02d.jpg", i, side[0],i, side[0],g_idx);

            cout << fr << endl << sc << endl << endl;
            string file = base + fr;
            string file2 = base + sc;

            string first = fr;
            string second = sc;

            if (!exists(file)) continue;
            if (!exists(file2)) continue;

            cout << fr << " " << sc << endl;

            if(max < max_count) return;
            max_count++;

            in.push_back(tuple<string, string, int>(first, second, i));
        }
    }

}
vector<Mat> images;
vector<Mat> maps;
vector<int> label;

void save_pres(string base) {
    vector<Mat>::iterator ii;  // declare an iterator to a vector of strings
    vector<Mat>::iterator im;  // declare an iterator to a vector of strings
    vector<int>::iterator il;  // declare an iterator to a vector of strings

    int i = 0;
    for(ii = images.begin(), im = maps.begin(), il = label.begin(); ii != images.end(), im != maps.end(), il != label.end(); ii++, im++, il++,i++ )    {
        string baseBase = base + "/" + std::to_string(i) + "_" + std::to_string((*il));

        string img = baseBase + "_f.jpg";
        string map = baseBase + "_m.jpg";

        imwrite(img, (*ii));
        imwrite(map, (*im));
    }
}
vector<int> hamDists;

void pre_process(im_pair in, string base, string mapBase) {

    string first_name = "";
    for (im_pair::const_iterator i = in.begin(); i != in.end(); ++i) {
        cout << get<0>(*i) << endl;
        cout << get<1>(*i) << endl;
        //cout << get<2>(*i) << endl;

        string file = base + get<0>(*i);
        string Mfile = mapBase + get<0>(*i) + "_m.jpg";

        string file2 = base + get<1>(*i);
        string Mfile2 = mapBase + get<1>(*i) + "_m.jpg";

        Mat original = imread(file, IMREAD_UNCHANGED);
        Mat original2 = imread(file2, IMREAD_UNCHANGED);
        Mat mapOrig = imread(Mfile, IMREAD_UNCHANGED);
        Mat mapOrig2 = imread(Mfile2, IMREAD_UNCHANGED);

        flatImage(original);
        flatImage(original2);
        flatImage(mapOrig);
        flatImage(mapOrig2);

        int min = INT_MAX;
        int min_ofset = 0;
        for(int ofs = 0; ofs < original.cols; ofs++) {
            int val = hammingDist(original, mapOrig, original2, mapOrig2, ofs);
            if(val < min) {
                min = val;
                min_ofset = ofs;
            }
        }
        //cout << "Min dist: " << min << " Ofset" << min_ofset << endl;
        hamDists.push_back(min);

        Mat fixed = cv::Mat::zeros(original.size(), CV_8U);
        Mat fixedMap = cv::Mat::zeros(original.size(), CV_8U);
        matFromOfset(original2, fixed, min_ofset);
        matFromOfset(mapOrig2, fixedMap, min_ofset);

        if(first_name != file) {
            images.push_back(original);
            maps.push_back(mapOrig);
            label.push_back(get<2>(*i));
            first_name = file;
            //cout << "saved first" << endl;
        }

        images.push_back(fixed);
        maps.push_back(fixedMap);
        label.push_back(get<2>(*i));
        //cout << "saved second" << endl;

        //imshow("F", original);
        //imshow("S", fixed);

        waitKey(0);
    }
    /*
     if ( std::find(vec.begin(), vec.end(), item) != vec.end() )
       do_this();
    else
       do_that();
     * */
}
Mat train_images;
Mat train_labels;

Mat test_images;
Mat test_labels;

int number_of_classes = 0;

void loadData(string base, float train_percentage = 0.7) {
    vector<Mat> obrazky;
    vector<Mat> mapy;
    vector<int> lable;
    set<int> classes;

    map<int, int> lab_map;


    for (const auto & entry : fs::directory_iterator(base)) {
        std::string cesta = entry.path().u8string();

        string pbase = cesta.substr(base.length(), cesta.length());

            string typ = pbase.substr(pbase.find_last_of("_"), 2);

        string sbase = pbase.substr(0, pbase.find_last_of("_"));

        int id = stoi(sbase.substr(0, sbase.find_last_of("_")));
        int clas = stoi(sbase.substr(sbase.find_last_of("_")+1, 50));
        //cout << sbase << " " << id << " " << clas << endl;

        if(typ == "_m") continue;

        /*std::string delimiter = ">=";
        std::string token = ); // token is "scott"
        */
        string mapa = base + "/" + to_string(id) + "_" + to_string(clas) + "_m.jpg";
        Mat im = imread(cesta);
        Mat map = imread(mapa);

        Mat out;
        flatImage(im);
        flatImage(map);

        im.copyTo(out, map);

        cv::resize(out, out, cv::Size(), 0.7, 0.7);
        //cv::resize(map, map, cv::Size(), 0.5, 0.5);


        //im.copyTo(im, map);

        //imshow("im", out);
        //imshow("mp", map);
        //waitKey(0);

        cv::normalize(im, im, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        obrazky.push_back(out);
        mapy.push_back(map);
        lable.push_back(clas);
        classes.insert(clas);
/*
        Mat out;
        cv::normalize(im, out, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        cout << typ << endl;
        */
    }

    vector<Mat>::iterator ii;  // declare an iterator to a vector of strings
    vector<Mat>::iterator im;  // declare an iterator to a vector of strings
    vector<int>::iterator il;  // declare an iterator to a vector of strings

    cout << classes.size() << endl;

    int i = 0;
    for(int f : classes) {

        lab_map.insert(pair<int, int>(f, i));
        i++;
        //cout << lab_map.at(f) << "a" << i  << "a" << f << endl;
    }

    number_of_classes = lab_map.size();
    int* per_class = new int[lab_map.size()];
    int* per_class_now = new int[lab_map.size()];
    for(int i =0; i < lab_map.size(); i++) {
        per_class[i] = 0;
        per_class_now[i] = 0;
    }
    for(il = lable.begin(); il != lable.end(); il++ ) {
        per_class[lab_map.at(*il)]++;
        //cout << per_class[lab_map.at(*il)] << endl;
    }
    for(ii = obrazky.begin(), im = mapy.begin(), il = lable.begin(); ii != obrazky.end(), im != mapy.end(), il != lable.end(); ii++, im++, il++ ) {
        Mat row = (*ii).reshape(1,1);
        int clas = lab_map.at(*il);
        cv::Mat zaradenie = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
        zaradenie.at<float>(clas) = 1;

        if(per_class_now[clas] < int(per_class[clas]*train_percentage)) {
            train_images.push_back(row);
            train_labels.push_back(zaradenie);
            per_class_now[clas]++;
        } else {
            test_images.push_back(row);
            test_labels.push_back(zaradenie);
        }
        //cout << train_images.rows << " " << train_images.cols << endl;
        //cout << test_images.size() << " " << test_labels.size() << endl << endl;

        /*imshow("s", (*ii));
        waitKey(0);
         */
    }

}

Mat flat_predicted(Mat predicted) {
    Mat out = cv::Mat::zeros(predicted.size(), CV_8U);

    for (int i = 0; i < predicted.rows; i++) {
        float max = -2;
        int idx = -1;
        for (int j = 0; j < predicted.cols; j++) {

            float val = predicted.at<float>(i, j);
            if(val > max) {
                max = val;
                idx = j;
            }
        }
        out.at<char>(i, idx) = 1;
    }
    return out;
}

float assert_predict(Mat predicted, Mat truth) {
    int corr = 0;
    int fail = 0;
    int total = predicted.rows;
    for (int i = 0; i < predicted.rows; i++) {
        int failed = 0;
        for (int j = 0; j < predicted.cols; j++) {

            char val = predicted.at<char>(i, j);
            float val2 = truth.at<float>(i, j);
            if((int)val != (int)val2) {
                failed = 1;
                break;
            }
        }
        if(failed) {
            fail++;
        } else {
            corr++;
        }

    }
    float usp = ((double)corr/total);
    cout << "Good: " << corr << " Bad:" << fail << " Celkovo" << total << " Uspenost: " << usp*100 << "%" << endl;
    return usp;
}

int main( int argc, const char** argv )
{
    const string originalBase = "../../iris_NEW/";
    //const string base = "../../iris_NEW_procesed/";
    const string base = "../../iris_NEW_3/";
    const string mapBase = "../../iris_NEW_procesed/";

    const string saveBase = "../../forth_4/";
    //const string saveBase2 = "../../forth_4/";


    im_pair correct_pairs;
    im_pair not_correct_pairs;
/*
    right_pairs(correct_pairs, base, 200, true);
    pre_process(correct_pairs, base, mapBase);

    for(auto s : hamDists) {
        cout << s << endl;
    }

    //pre_process(im_pair in, string base, string mapBase)
    //right_pairs(not_correct_pairs, base, 200, true);

    //pre_process(correct_pairs, base, mapBase);

    //save_pres(saveBase2);

    return 0;
*/

    //loadData(saveBase);

    parseCSV(originalBase);
    std::vector<std::vector<std::string> >::iterator row;
    std::vector<std::string>::iterator col;
    int cnt = 0;
    vector<Mat> buf;
    int last_l = 0;
    char last_lr = 0;
    int classes = 7;
    for (row = parsedCsv.begin(); row != parsedCsv.end(); row++, cnt++) {
        std::vector<std::string> data = *row;
        //row++;
        //std::vector<std::string> data2 = *row;
        //flatten(base, mapBase, data, data2, 365, 60);
        int lab = 0;
        char lr = 0;

        Mat desc = sift_Extract(mapBase, data, 20, &lab, &lr);
        if(lab > classes) break;
        if(lr != 0) {
            if(cnt > 0 && last_lr != lr) {
                last_lr = lr;
                int riad=0;
                //cout << "Pocet: " << buf.size() << endl;

                cv::Mat zaradenie = cv::Mat::zeros(cv::Size(classes, 1), CV_32F);
                zaradenie.at<float>(lab-1) = 1;

                for(Mat mt : buf) {
                    cout << mt.rows << " " << mt.cols << endl;
                    if(int(buf.size()*0.7) > riad){
                   //     cout << "Train" << endl;
                        train_images.push_back(mt);
                        train_labels.push_back(zaradenie);
                    } else {
                        test_images.push_back(mt);
                        test_labels.push_back(zaradenie);
                     //   cout << "Test" << endl;
                    }
                    //cout << zaradenie << endl;
                    riad++;
                }
                buf.clear();
            }
            buf.push_back(desc);
        }
        cout << "\n";
    }

    cv::normalize(train_images, train_images, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(test_images, test_images, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    //cout << test_images << endl;
    train_images.convertTo(train_images, CV_32F);
    train_labels.convertTo(train_labels, CV_32F);
    test_images.convertTo(test_images, CV_32F);
    test_labels.convertTo(test_labels, CV_32F);

    cout << "Train imgs: " << train_images.rows << endl;
    cout << "Test imgs: " << test_images.rows << endl;

    int networkInputSize = train_images.cols;
    int networkOutputSize = train_labels.cols;
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();

    cout << number_of_classes << endl;
    std::vector<int> layerSizes = { networkInputSize, 70, 40,
                                    networkOutputSize };

    mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1, 0));
    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(train_images,cv::ml::ROW_SAMPLE,train_labels,cv::Mat(),cv::Mat(),cv::Mat(),cv::Mat());

    mlp->setLayerSizes(layerSizes);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);

    //mlp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 900, 0.00001));
    mlp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001);

    cout << "tu" << endl;

    mlp->train(trainData);
    cv::Mat predictions;
    int maxEpoch = 200;
    float last_succ = 0;
    int poor_epochs = 0;

    // Settings
    int poor_epochs_stop = 50;
    float poor_epoch_tresh = 0.0001;

    vector<float> train_succ;
    vector<float> test_succ;

    for(int nEpochs = 2; nEpochs <= maxEpoch; nEpochs++) {
        // train network1 with one more epoch
        mlp->train(trainData,cv::ml::ANN_MLP::UPDATE_WEIGHTS);

        //if(nEpochs % 10 == 0) {
            cout << "Iteration: " << nEpochs << " / " << maxEpoch << endl;
            mlp->predict(train_images, predictions);

            Mat prd = flat_predicted(predictions);

            float train_s = assert_predict(prd, train_labels);
            train_succ.push_back(train_s);

            mlp->predict(test_images, predictions);

            prd.release();
            prd = flat_predicted(predictions);

            float suc = assert_predict(prd, test_labels);
            float dif = (suc - last_succ);
            cout << "Diff" << dif << endl;
            test_succ.push_back(suc);
            last_succ = suc;

            if(dif < poor_epoch_tresh) {
                poor_epochs++;
            } else {
                poor_epochs = 0;
            }

            if(poor_epochs >= poor_epochs_stop) {
                cout << "TRAIN STOP" << endl;
                break;
            }

            cout << endl;

    }

    mlp->predict(test_images, predictions);

    Mat prd = flat_predicted(predictions);

    assert_predict(prd, test_labels);

    mlp->save("nn.yml");

    int i = 2;
    for(auto s : train_succ) {
        cout << i << "\t" << s << endl;
        i++;
    }
    i = 2;
    cout << endl << endl;
    for(auto s : test_succ) {
        cout << i << "\t" << s << endl;
        i++;
    }


    return 0;
    // Correct pairs
    right_pairs(correct_pairs, base, 20);
/*
    cout << get<0>(correct_pairs.at(22)) << endl;
    cout << get<1>(correct_pairs.at(22)) << endl;
*/
    pre_process(correct_pairs, base, mapBase);

    save_pres(saveBase);

    /*
    parseCSV(originalBase);
    std::vector<std::vector<std::string> >::iterator row;
    std::vector<std::string>::iterator col;
    int cnt = 0;
    for (row = parsedCsv.begin(); row != parsedCsv.end(); row++, cnt++) {
        std::vector<std::string> data = *row;
        //row++;
        //std::vector<std::string> data2 = *row;
        //flatten(base, mapBase, data, data2, 365, 60);
        sift_Extract(mapBase, data);
        cout << "\n";
    }
*/
    return 0;
}
