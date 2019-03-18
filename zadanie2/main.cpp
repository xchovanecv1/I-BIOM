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

    /*
    int slope = (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);
     */
}

double flatten(string base, std::vector<std::string> data, int width, int height) {

    string file = base + data.at(0);
    if (!exists(file)) return -1;

    cout << data.at(0) << "test";

    int x_offset = 100, y_offset = 200;

    Mat original = imread(file, IMREAD_GRAYSCALE);


    Mat resized(original.rows + (y_offset * 2), original.cols + (x_offset * 2), CV_8UC1);

    original.copyTo(resized(Rect((x_offset), (y_offset), original.cols, original.rows)));

    original = resized;

    Mat draw = original.clone();
    imshow("Original", original);

    // Zrenicka
    int zrenicka_x = x_offset+stoi(data.at(1)), zrenicka_y = y_offset+stoi(data.at(2)), zrenicka_radius = stoi(data.at(3));
    Point center_zrenicka = Point(zrenicka_x, zrenicka_y);
    Size zrenicka_axes( zrenicka_radius, zrenicka_radius );
    vector<Point> zrenicka_points;
    ellipse2Poly( center_zrenicka, zrenicka_axes, 0, 0, 360, 1, zrenicka_points );

    circle(draw, center_zrenicka, 1, Scalar(0, 100, 100), 1, LINE_AA);
    circle(draw, center_zrenicka, zrenicka_radius, Scalar(255, 0, 255), 1, LINE_AA);

    // Duhovka
    int duhovka_x = x_offset+stoi(data.at(4)), duhovka_y = y_offset+stoi(data.at(5)), duhovka_radius = stoi(data.at(6));
    Point center_duhovka = Point(duhovka_x, duhovka_y);
    Size duhovka_axes( duhovka_radius, duhovka_radius );
    vector<Point> duhovka_points;
    ellipse2Poly( center_duhovka, duhovka_axes, 0, 0, 360, 1, duhovka_points );

    circle(draw, center_duhovka, 1, Scalar(0, 100, 100), 1, LINE_AA);
    circle(draw, center_duhovka, duhovka_radius, Scalar(0, 0, 0), 1, LINE_AA);

    //Horne v
    int dolnev_x = x_offset+stoi(data.at(7)), dolnev_y = y_offset+stoi(data.at(8)), dolnev_radius = stoi(data.at(9));
    Point dolnev_center = Point(dolnev_x, dolnev_y);

    //Dolne v
    int hornev_x = x_offset+stoi(data.at(10)), hornev_y = y_offset+stoi(data.at(11)), hornev_radius = stoi(data.at(12));
    Point hornev_center = Point(hornev_x, hornev_y);

    Mat dolne_map_mask = Mat::zeros(original.rows, original.cols, CV_8UC1);
    Mat horne_map_mask = Mat::zeros(original.rows, original.cols, CV_8UC1);


    circle(dolne_map_mask, center_duhovka,duhovka_radius, Scalar(128,128,128), -1, 8,0);
    circle(dolne_map_mask, dolnev_center,dolnev_radius, Scalar(255,255,255), -1, 8,0);

    circle(horne_map_mask, center_duhovka,duhovka_radius, Scalar(128,128,128), -1, 8,0);
    circle(horne_map_mask, hornev_center, hornev_radius, Scalar(255,255,255), -1, 8,0);

    imshow("Dolne Mask map", dolne_map_mask);
    imshow("Horne Mask map", horne_map_mask);



    int both_max_x = max(duhovka_points.size(), zrenicka_points.size());

    both_max_x = width;
    int both_max_y = height;

    vector<vector<Point>> pts;

    int image_height = 0;
    int image_width = max(duhovka_points.size(), zrenicka_points.size());

    int pic_h = 0;

    int i = 0;
    for(i = 0; i < both_max_x; i++){

        int duhovka_idx = mapRange(i,0,both_max_x-1,0,duhovka_points.size()-1);
        int zrenicka_idx = mapRange(i,0,both_max_x-1,0,zrenicka_points.size()-1);

        Point duhovka_point = duhovka_points[duhovka_idx];
        Point zrenicka_point = zrenicka_points[zrenicka_idx];

        circle(draw, duhovka_point, 1, Scalar(255, 0, 255), 1, LINE_AA);
        circle(draw, zrenicka_point, 1, Scalar(255, 0, 255), 1, LINE_AA);


        LineIterator it(original, zrenicka_point, duhovka_point, 8);

        vector<Point> buf(it.count);

        int d = 0;
        for(d = 0; d < it.count; d++, ++it) {
            buf[d] = it.pos();
        }
        image_height = it.count;

        vector<Point> pxs;

        for(d = 0; d < both_max_y; d++) {
            int point_idx = mapRange(d,0,both_max_y-1,0,buf.size()-1);
            pxs.push_back(buf[point_idx]);

        }
        pic_h = max(pic_h, (int)pxs.size());

        pts.push_back(pxs);

    }

    Mat final = Mat::zeros(height, width, CV_8UC1);
    Mat final_map = Mat::zeros(height, width, CV_8UC1);
    Mat final_map_inverse = Mat::zeros(height, width, CV_8UC1);

    for(int i=0; i < pts.size(); i++) {
        for(int j = 0; j < pts[i].size(); j++) {
            Point before;
            if(i == 0) {
                before = pts[pts.size()-2][j];
            } else {
                before = pts[i-1][j];
            }
            Point after;
            if(i == pts.size()-1) {
                after = pts[0][j];
            } else {
                after = pts[i+1][j];
            }

            Point actual = pts[i][j];

            int final_pixel = original.at<uchar>(actual);
            int final_pixel_count = 1;
            if(width < image_width) {
                int r = euclidDist(actual,center_zrenicka);

                Mat cutout = Mat::zeros(original.rows, original.cols, CV_8UC1);

                circle(cutout, center_zrenicka, r, Scalar(255, 0, 255), 1, LINE_AA);
                circle(cutout, actual, 1, Scalar(128, 0, 255), 1, LINE_AA);
                circle(cutout, after, 1, Scalar(128, 0, 255), 1, LINE_AA);

                Size axes( r, r );
                vector<Point> c_points;
                ellipse2Poly( center_zrenicka, axes, 0, 0, 360, 1, c_points );

                vector<Point> first;
                vector<Point> second;
                int found=0;
                for(int c=0; c < c_points.size(); c++) {
                    int d_f = euclidDist(actual,c_points[c]);
                    int d_s = euclidDist(after,c_points[c]);

                    Scalar colour_h = cutout.at<uchar>(c_points[c]);
                    if(d_f == 0 || d_s == 0) {
                        found++;
                    }
                    if(found == 0 || found == 2) {
                        first.push_back(c_points[c]);
                    }
                    if(found == 1) {
                        second.push_back(c_points[c]);
                    }
                }

                vector<Point> *final_pv;

                if(first.size() <= second.size()) {
                    final_pv = &first;
                } else {
                    final_pv = &second;
                }

                for(int f=0; f < final_pv->size(); f++) {
                    final_pixel += original.at<uchar>((*final_pv)[f]);
                    final_pixel_count++;
                }
                cutout.release();
            }

            if(height < image_height) {
                if(j != 0) {
                    before = pts[i][j-1];

                    LineIterator itt(original, actual, before, 8);

                    int d = 0;
                    for(d = 0; d < itt.count; d++, ++itt) {
                        final_pixel += original.at<uchar>(itt.pos());
                        final_pixel_count++;
                    }
                }

            }

            uchar final_p = (uchar)(final_pixel/final_pixel_count);
            final.at<uchar>(j, i) = final_p;


            Scalar colour_h = horne_map_mask.at<uchar>(actual);
            Scalar colour_d = dolne_map_mask.at<uchar>(actual);

            if(colour_d.val[0] == 128 || colour_h.val[0] == 128) {
                final_map.at<uchar>(j, i) = 255;
            } else {
                final_map_inverse.at<uchar>(j, i) = 255;
            }

            circle(draw, actual, 1, Scalar(255, 0, 255), 1, LINE_AA);
            circle(draw, before, 1, Scalar(0, 0, 255), 1, LINE_AA);
            circle(draw, after, 1, Scalar(0, 0, 255), 1, LINE_AA);


        }
    }

    Mat final_i;
    Mat final_map_i;
    Mat final_map_inverse_i;

    Mat final_image;

    cv::flip(final, final_i, 0);
    cv::flip(final_map, final_map_i, 0);
    cv::flip(final_map_inverse, final_map_inverse_i, 0);

    imshow("final", final_i);
    imshow("final map", final_map);
    imshow("final map inverse", final_map_inverse_i);

    final_i.copyTo(final_image, final_map_inverse_i);
    imshow("RESULT", final_image);


    imshow("Working", draw);

    // FREE MEMORY
    original.release();
    draw.release();
    final.release();
    final_map.release();
    final_map_inverse.release();

    final_i.release();
    final_map_i.release();
    final_map_inverse_i.release();

    dolne_map_mask.release();
    horne_map_mask.release();

    waitKey(0);

}

int main( int argc, const char** argv )
{
    const string base = "../../iris_NEW/";
    parseCSV(base);
    std::vector<std::vector<std::string> >::iterator row;
    std::vector<std::string>::iterator col;
    int cnt = 0;
    for (row = parsedCsv.begin(); row != parsedCsv.end(); row++, cnt++) {
        flatten(base, *row, 800, 600);

        cout << "\n";
    }

    return 0;
}
