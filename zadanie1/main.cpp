#include "opencv2/opencv.hpp"
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
using namespace cv;

std::vector<std::vector<std::string> > parsedCsv;

struct Circle {
    int x;
    int y;
    double r;
};

Circle createCircle(int x, int y, double radius) {
    Circle* bf = new Circle;
    bf->x = x;
    bf->y = y;
    bf->r = radius;

    return *bf;
}

double intersectionArea(Circle A, Circle B) {

    double d = hypot(B.x - A.x, B.y - A.y);

    if (d < A.r + B.r) {

        double a = A.r * A.r;
        double b = B.r * B.r;

        double x = (a - b + d * d) / (2 * d);
        double z = x * x;
        double y = sqrt(a - z);

        if (d < abs(B.r - A.r)) {
            return M_PI * min(a, b);
        }
        return a * asin(y / A.r) + b * asin(y / B.r) - y * (x + sqrt(z + b - a));
    }
    return 0;
}

double unionArea(Circle A, Circle B) {
    double ret = (M_PI * A.r * A.r) + (M_PI * B.r * B.r) - intersectionArea(A, B);
    return ret;
}

double uspesnost (Circle A, Circle B) {
    return (intersectionArea(A, B) / unionArea(A, B));
}

int findFile(string name) {

    std::vector<std::vector<std::string> >::iterator row;
    for (row = parsedCsv.begin(); row != parsedCsv.end(); row++) {
        string bf = *(row->begin());
        std::size_t found = bf.find_last_of("/\\");
        string file = bf.substr(found+1);

        if(file.compare(name) == 0) {
            return row - parsedCsv.begin();
        }
    }

    return -1;
}

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
    /*for (row = parsedCsv.begin(); row != parsedCsv.end(); row++) {
        for (col = row->begin(); col != row->end(); col++) {
            cout << *col << " ";
        }
        cout << "\n";
    }*/
};
bool exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

double euclidDist(double x1, double y1, double x2, double y2)
{
    double x = x1 - x2;
    double y = y1 - y2;
    double dist;

    dist = pow(x, 2) + pow(y, 2);
    dist = sqrt(dist);

    return dist;
}

double  zrenicka_u  = 0,
        duhovka_u   = 0,
        horne_v_u   = 0,
        dolne_v_u   = 0;
int     uspesnost_p = 0,
        uspesnost_p_celkovo = 0;

bool show_im = false;


void calcSucc() {

    double zrenicka_u_c  = zrenicka_u / uspesnost_p;
    double duhovka_u_c   = duhovka_u / uspesnost_p;
    double horne_v_u_c   = horne_v_u / uspesnost_p;
    double dolne_v_u_c   = dolne_v_u / uspesnost_p;

    cout << "Uspesne Celkovo zrenicky: " << zrenicka_u_c << "\n";
    cout << "Uspesne Celkovo duhovka: " << duhovka_u_c << "\n";
    cout << "Uspesne Celkovo horne vieco: " << horne_v_u_c << "\n";
    cout << "Uspesne Celkovo dolne viecko: " << dolne_v_u_c << "\n";

    cout << "Uspesne Celkovo: " << (zrenicka_u_c+duhovka_u_c+horne_v_u_c+dolne_v_u_c)/4 << "\n";

    cout << "Celkovo Celkovo zrenicky: " << zrenicka_u / uspesnost_p_celkovo<< "\n";
    cout << "Celkovo Celkovo duhovka: " << duhovka_u  / uspesnost_p_celkovo<< "\n";
    cout << "Celkovo Celkovo horne vieco: " << horne_v_u / uspesnost_p_celkovo << "\n";
    cout << "Celkovo Celkovo dolne viecko: " << dolne_v_u / uspesnost_p_celkovo << "\n";

    cout << "Celkovo Celkovo: " << (zrenicka_u+duhovka_u+horne_v_u+dolne_v_u)/ uspesnost_p_celkovo/4 << "\n";
}


double doIris(string base, std::vector<std::string> data) {

    double  zrenicka_b  = 0,
            duhovka_b   = 0,
            horne_v_b   = 0,
            dolne_v_b   = 0;

    string file = base + data.at(0);
    if(!exists(file)) return -1;

    cout << data.at(0) << "test";

    int x_offset = 100, y_offset = 200;

    Mat original = imread(file, IMREAD_GRAYSCALE);


    Mat resized(original.rows + (y_offset * 2), original.cols + (x_offset * 2), CV_8UC1);

    original.copyTo( resized( Rect((x_offset), (y_offset), original.cols, original.rows) ) );
    //imshow("res", resized);

    original = resized;

    Mat draw = original.clone();
    if(show_im) imshow("Original", original);

    Mat iris_canny, lid_preprocc;
    vector<Vec3f> circless;
    Canny(original, iris_canny, 180, 100 , 3);
    if(show_im) imshow("Cannied iris", iris_canny);
    //lid_preprocc = lid_canny;
    //lid_canny = blacked_pupil;
    GaussianBlur(original, lid_preprocc, Size(5,5), 1.7, 0);
    //lid_preprocc = original;
    if(show_im) { imshow("paster", lid_preprocc); waitKey(0); }
    /// EYE DETECTION*/
    HoughCircles(lid_preprocc, circless, HOUGH_GRADIENT, 1,
                 lid_preprocc.rows/2,
                 150, 50, 1, 120
    );
    cout << circless.size();
    Point zrenicka_cent;
    Vec3i zrenicka_dat;
    for( size_t i = 0; i < circless.size(); i++ )
    {
        if(i > 0) break;
        Vec3i c = circless[i];
            Point center = Point(c[0], c[1]);
            zrenicka_cent = center;
            // circle center
            circle( draw, center, 1, Scalar(0,255,255), 3, LINE_AA);
            // circle outline
            int radius = c[2];
            circle( draw, center, radius, Scalar(0,128,255), 3, LINE_AA);
    }
    if(circless.size()){

        int r_x = stoi(data.at(1)), r_y = stoi(data.at(2)), r_r = stoi(data.at(3));
        int g_x = circless[0][0], g_y = circless[0][1], g_r = circless[0][2];
        zrenicka_dat = circless[0];
        r_x += x_offset; r_y += y_offset;

        Circle rajt = createCircle(r_x, r_y, r_r);
        Circle fcirc = createCircle(g_x, g_y, g_r);
        cout << "\nUspesnost zrenicky: " << uspesnost(fcirc, rajt) << "\n";

        Rect2d r1(r_x - r_r, r_y - r_r, r_r * 2, r_r * 2);
        Rect2d r2(g_x - g_r, g_y - g_r, g_r * 2, g_r * 2);
        Rect2d r3 = r1 & r2;
        Rect2d r4 = r1 | r2;

        zrenicka_b = r3.area() / r4.area();
        cout << "\nUspesnost zrenicky: " << zrenicka_b << "\n";

        //return uspesnost(fcirc, rajt);
    } else {
        uspesnost_p_celkovo++;
        return 0;
    }
    /// Apply Histogram Equalization
    equalizeHist( lid_preprocc, lid_preprocc );
   // GaussianBlur(lid_preprocc, lid_preprocc, Size(5,5), 1.7, 0);

    if(show_im) imshow("hist", lid_preprocc);

    int low = 100, high = 10, min_r = zrenicka_dat[2]*1.2, max_r = 120;
    Canny(lid_preprocc, iris_canny, low, high , 3);
    if(show_im) imshow("Cannied outer", iris_canny);

    vector<Vec3f> circ_out;
    HoughCircles(lid_preprocc, circ_out, HOUGH_GRADIENT, 2,
                 5,
                 low, high, min_r, max_r
    );
    //cout << "Outer:" << circ_out.size();
    double min = std::numeric_limits<double>::max();
    int min_pos = 0;
    for( size_t i = 0; i < circ_out.size(); i++ )
    {
        Vec3i c = circ_out[i];
        if(c[0] > original.cols || c[1] > original.rows) continue;
        Point center = Point(c[0], c[1]);
        double dist = norm(zrenicka_cent - center);
        if(dist <= min) {
            min = dist;
            min_pos = i;
        }
    }
    Vec3i zrenicka_c;
    double outer_radius = 0;
    if(circ_out.size()){
        Vec3i c = circ_out[min_pos];
        zrenicka_c = c;
        Point center = Point(c[0], c[1]);
        outer_radius = c[2];

        int r_x = stoi(data.at(4)), r_y = stoi(data.at(5)), r_r = stoi(data.at(6));
        int g_x = c[0], g_y = c[1], g_r = c[2];
        r_x += x_offset; r_y += y_offset;

        Circle rajt = createCircle(r_x, r_y, r_r);
        Circle fcirc = createCircle(g_x, g_y, g_r);
        cout << g_x << " " << g_y << " " << g_r << " \n";
        cout << r_x << " " << r_y << " " << r_r << " \n";
        cout << "\nUspesnost duhovky: " << uspesnost(fcirc, rajt) << "\n";


        Rect2d r1(r_x - r_r, r_y - r_r, r_r * 2, r_r * 2);
        Rect2d r2(g_x - g_r, g_y - g_r, g_r * 2, g_r * 2);
        Rect2d r3 = r1 & r2;
        Rect2d r4 = r1 | r2;

        duhovka_b = r3.area() / r4.area();
        cout << "\nUspesnost duhovky: " << duhovka_b << "\n";


        circle( draw, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( draw, center, radius, Scalar(255,0,255), 3, LINE_AA);
    } else {
        uspesnost_p_celkovo++;
        return 0;
    }
    if(show_im) {imshow("Duhovka", draw); waitKey(0);}
    // viecka
    GaussianBlur(original, lid_preprocc, Size(5,5), 1.7, 0);

    low = 140, high = 5;
    Canny(lid_preprocc, iris_canny, low, high , 3);
    if(show_im) imshow("Cannied outer", iris_canny);
    int rad_min = zrenicka_c[2] * 2, rad_max = zrenicka_c[2]* 4;
    vector<Vec3f> circ_vie;
    HoughCircles(lid_preprocc, circ_vie, HOUGH_GRADIENT, 2,
                 80,
                 low, high, rad_min, rad_max
    );
    cout << "viecka:" << circ_vie.size();
    Vec3f horne_v_mean, dolne_v_mean;
    int horne_v_mean_c = 0, dolne_v_mean_c = 0;

    for( size_t i = 0; i < circ_vie.size(); i++ )
    {
        Vec3i c = circ_vie[i];
        Point center = Point(c[0], c[1]);

        // horna
        if(c[1] > zrenicka_c[1]) {

            double dist = abs(zrenicka_c[0] - c[0]);
            if(dist < 20) {
                horne_v_mean[0] += c[0];
                horne_v_mean[1] += c[1];
                horne_v_mean[2] += c[2];
                horne_v_mean_c++;
            }
        }
    }
    Point horne_viecko_center;
    Vec3i horne_viecko_vec;
    if(horne_v_mean_c) {
        Vec3i c;
        c[0] += horne_v_mean[0] / horne_v_mean_c;
        c[1] += horne_v_mean[1] / horne_v_mean_c;
        c[2] += horne_v_mean[2] / horne_v_mean_c;
        horne_viecko_vec = c;
        Point center = Point(c[0], c[1]);
        horne_viecko_center = center;
        Circle rajt = createCircle(stoi(data.at(7)), stoi(data.at(8)), stoi(data.at(9)));
        Circle fcirc = createCircle(circless[0][0], circless[0][1], circless[0][2]);
        //cout << "\n" << uspesnost(fcirc, rajt) << "\n";

        int r_x = stoi(data.at(10)), r_y = stoi(data.at(11)), r_r = stoi(data.at(12));
        int g_x = c[0], g_y = c[1], g_r = c[2];
        r_x += x_offset; r_y += y_offset;

        Rect2d r1(r_x - r_r, r_y - r_r, r_r * 2, r_r * 2);
        Rect2d r2(g_x - g_r, g_y - g_r, g_r * 2, g_r * 2);
        Rect2d r3 = r1 & r2;
        Rect2d r4 = r1 | r2;

        horne_v_b = r3.area() / r4.area();
        cout << "\nUspesnost horneho viecka: " << horne_v_b << "\n";


        circle(draw, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(draw, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
    } else {
        uspesnost_p_celkovo++;
        return 0;
    }
    horne_viecko_center.y += 50;
    Mat mask = Mat::zeros(lid_preprocc.rows, lid_preprocc.cols, CV_8UC1);
    circle(mask, horne_viecko_center, horne_viecko_vec[2], Scalar(255,255,255), -1, 8, 0 );
    Mat final_result = Mat::zeros(lid_preprocc.rows, lid_preprocc.cols, CV_8UC1);
    lid_preprocc.copyTo(final_result, mask);

    if(show_im) imshow("final orez", final_result);
    low = 140, high = 5;
    Canny(final_result, iris_canny, low, high , 3);
    if(show_im) imshow("Cannied outer", iris_canny);
    rad_min = zrenicka_c[2] * 2, rad_max = zrenicka_c[2]* 8;
    vector<Vec3f> circ_vie_d;
    HoughCircles(final_result, circ_vie_d, HOUGH_GRADIENT, 2,
                 80,
                 low, high, rad_min, rad_max
    );
    cout << "viecka:" << circ_vie_d.size();

    for( size_t i = 0; i < circ_vie_d.size(); i++ )
    {
        Vec3i c = circ_vie_d[i];
        Point center = Point(c[0], c[1]);

        if(c[1] < zrenicka_c[1]) {

            double dist = abs(zrenicka_c[0] - c[0]);
            if(dist < 80 && c[1] + c[2] > zrenicka_c[1]){
                dolne_v_mean[0] += c[0];
                dolne_v_mean[1] += c[1];
                dolne_v_mean[2] += c[2];
                dolne_v_mean_c++;

            }
        }
    }

    if(dolne_v_mean_c){
        Vec3i c;
        c[0] += dolne_v_mean[0] / dolne_v_mean_c;
        c[1] += dolne_v_mean[1] / dolne_v_mean_c;
        c[2] += dolne_v_mean[2] / dolne_v_mean_c;
        Point center = Point(c[0], c[1]);

        int r_x = stoi(data.at(7)), r_y = stoi(data.at(8)), r_r = stoi(data.at(9));
        int g_x = c[0], g_y = c[1], g_r = c[2];
        r_x += x_offset; r_y += y_offset;

        Rect2d r1(r_x - r_r, r_y - r_r, r_r * 2, r_r * 2);
        Rect2d r2(g_x - g_r, g_y - g_r, g_r * 2, g_r * 2);
        Rect2d r3 = r1 & r2;
        Rect2d r4 = r1 | r2;

        dolne_v_b = r3.area() / r4.area();

        cout << "\nUspesnost dolneho  viecka: " << dolne_v_b << "\n";

        circle( draw, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( draw, center, radius, Scalar(0,0,0), 3, LINE_AA);
    } else {
        uspesnost_p_celkovo++;
        return 0;
    }

    zrenicka_u  += zrenicka_b;
    duhovka_u   +=  duhovka_b;
    horne_v_u   += horne_v_b;
    dolne_v_u   += dolne_v_b;

    uspesnost_p++;


    if(show_im) imshow("detected circless", draw);

    Mat mask_zr = Mat::zeros(original.rows, original.cols, CV_8UC1);
    circle(mask_zr, Point(zrenicka_c[0], zrenicka_c[1]), zrenicka_c[2], Scalar(255,255,255), FILLED);
    Mat fin_result = Mat::zeros(original.rows, original.cols, CV_8UC1);
    original.copyTo(fin_result, mask_zr);

    circle(fin_result, Point(zrenicka_c[0], zrenicka_c[1]), zrenicka_dat[2], Scalar(0,0,0), FILLED);
    if(show_im) imshow("Vysek duhovky", fin_result); waitKey(0);

    vector< vector<Point> > contours;
    int radius = zrenicka_c[2];
    int x = int(zrenicka_c[0] - radius);
    int y = int(zrenicka_c[1] - radius);
    // Add 2 elements to avoid information of the iris to be cut due to rounding errors
    int w = int(radius * 2) + 2;
    int h = w;
    Mat cropped_region = final_result( Rect(x,y,w,h) ).clone();
    // Now perform the unwrapping
    // This is done by the logpolar function who does Logpolar to Cartesian coordinates, so that it can get unwrapped properly
    Mat unwrapped;
    Point2f center (float(cropped_region.cols/2.0), float(cropped_region.cols /2.0));
    logPolar(cropped_region, unwrapped, center, 40, INTER_LINEAR +  WARP_FILL_OUTLIERS);
    imshow("unwrapped image polar", unwrapped);
    // Make sure that we only get the region of interest
    // We do not need the black area for comparing
    Mat thresholded;
    // Apply some thresholding so that you keep a white blob where the eye pixels are
    threshold(unwrapped, thresholded, 10, 255, THRESH_BINARY);
    imshow("thresholded ", thresholded); waitKey(0);
    // Run a contour finding algorithm to locate the iris pixels
    // Then define the bounding box
    findContours(thresholded.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // Use the bounding box as the ROI for cutting off the black parts
    Rect ROI = boundingRect(contours[0]);
    Mat iris_pixels = unwrapped(ROI).clone();
    imshow("iris pixels", iris_pixels);

    imshow("Vysek duhovky", original); waitKey(0);

    original.release();
    draw.release();
    final_result.release();
    lid_preprocc.release();
    iris_canny.release();
    mask.release();
    mask_zr.release();
    fin_result.release();

    calcSucc();


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
        doIris(base, *row);

        cout << "\n";
    }

    calcSucc();

    return 0;
}
