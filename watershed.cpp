#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cstdio>
#include <iostream>
#include <set>
#include "Transform.cpp"

using namespace cv;
using namespace std;

int alpha = 2 , beta = 155 , c = 7 , s = 1 , s1 = 18;

void sharp_image(Mat img, Mat& output)
{
    // do the laplacian filtering as it is
    // well we need to convert everything in something more deeper then CV_8U
    // because the kernel hsas some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    
    Mat imgLaplacian, sharp1, imgResult;
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1);

    filter2D(img, imgLaplacian, CV_32F, kernel);
    img.convertTo(sharp1, CV_32F);
    imgResult = sharp1 - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    output = imgResult; // copy back
    
}
void border(Mat& img)
{
    for (int i = 0; i < img.rows; ++i)
    {
        img.at<uchar>(i, 0) = img.at<uchar>(i,img.cols -1) = 0; 
    }

    for (int i = 0; i < img.cols; ++i)
    {
        img.at<uchar>(0,i) = img.at<uchar>(img.rows - 1, i) = 0;
    }
}

void get_markers(Mat img, Mat& output, int s1, int c, int alpha, int beta, int s, int& size, int flag)
{
    Mat bw,bw1,bw2;
    Mat src_copy = img.clone();
    cvtColor(img, bw1, CV_BGR2GRAY);
       
    medianBlur(bw1,bw1,3);
    imshow("q" , bw1);
    imwrite("bn1.jpg",bw1);
    bilateralFilter(bw1, bw, 2*alpha + 1, beta, 50 );
    
    imshow("Binary Image1", bw);
    imwrite("a.jpg" , bw);
    // threshold(bw, bw2, 40, 255, CV_THRESH_BINARY_INV| CV_THRESH_OTSU);
    adaptiveThreshold(bw,  bw2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 2*s1 + 1, c);
    imshow("Binary Image", bw2);
    imwrite("a1.jpg" , bw2);
    // Perform the distance transform algorithm
    Mat dist(img.rows, img.cols, CV_8UC1,Scalar(0));
    //distanceTransform(bw2,dist, CV_DIST_L2, 5);
    trans::distanceTransform(bw2,bw1,dist, CV_DIST_L2, 5,s);
    //  Normalize the distance image for range = {0.0, 1.0}
    //so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    imwrite("Distancet.jpg", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    if(flag)
        dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    imwrite("Peaks.jpg", dist);
   
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    //Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
       if (contourArea(contours[i]) > (img.rows*img.cols/8000))
        {
            drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
            drawContours(src_copy, contours, static_cast<int>(i), Scalar(0,0,255), -1);

        }
    }
    imwrite("markers.jpg", src_copy);
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    output = markers;
    size = contours.size();
    //Draw the background marker
    
    imshow("Marker", src_copy);
   
}

void _watershed(Mat src, Mat markers, Mat& output, int size, set<int>& index_arr)
{
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
   
    vector<Vec3b> colors;
    for (size_t i = 0; i < size; i++)
    {
        int b = 13 + 2*i;// theRNG().uniform(0, 255);
        int g =  13 + 3*i;//theRNG().uniform(0, 255);
        int r =  13 + 4*i;//theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
   

    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(size))
            {
                dst.at<Vec3b>(i,j) = colors[index-1];
                index_arr.insert(index-1);
               
            }

            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    // Visualize the final image
    output = dst;
    imshow("Final Result", dst);
    imwrite("final.jpg", dst);
    waitKey(0);
   
}

void get_small_segments(Mat dst, vector<Vec3b> colors,  set<int> index_arr, vector<vector<Point> >& contours)
{

    Mat bw3(dst.rows, dst.cols, CV_8UC3, Scalar(0,0,0));
    Mat output(dst.rows, dst.cols, CV_8UC3, Scalar(0,0,0));
    Mat edge;
    set<int>::iterator it;

    for (it = index_arr.begin(); it != index_arr.end(); ++it)
    {   
        vector<vector<Point> > contours_tmp;
        Scalar color( rand()&255, rand()&255, rand()&255 );
        int idx = *it;
       
        printf("%d %d\n", idx);
        inRange(dst, Scalar(colors[idx][0] ,colors[idx][1], colors[idx][2]), Scalar(colors[idx][0] + 2,colors[idx][1] + 2, colors[idx][2] + 2), bw3);

        Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
        dilate(bw3, bw3, kernel1);
        border(bw3);
        // imshow("edge", bw3);      
        // imshow("edge1", bw3);
        imwrite("edge1.jpg", bw3);
        findContours(bw3, contours_tmp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        contours.push_back(contours_tmp[0]);
        drawContours(output, contours_tmp, 0,color, -1);
        //imshow("dst", output);
        
        
   

        
    }
}



int main(int, char** argv)
{
    // Load the image
    Mat temp = imread(argv[1]);
    
    Mat src = temp.clone();
    // Check if everything was fine
    if (!temp.data)
        return -1;
    
    imshow("Black Background Image", src);
    // Create a kernel that we will use for accuting/sharpening our image4
    Mat bw,bw1,bw2;
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); // an approximation of second derivative, a quite strong kernel


    
   /* Mat imgLaplacian;
   */
    sharp_image(temp , src);
    imwrite("sharp.jpg", src);
    imshow("a" , src);
    waitKey(0);
    //Create binary image from source image
    
  
    // namedWindow("Object Detection",WINDOW_AUTOSIZE);
    // createTrackbar("alpha","Object Detection", &alpha, 50);
    // createTrackbar("beta","Object Detection", &beta, 180);
    // createTrackbar("gradient","Object Detection", &s, 100);
    // createTrackbar("constant","Object Detection", &c, 180);
    // createTrackbar("size","Object Detection", &s1, 180);

    
    Mat src_copy = temp.clone();
    vector<vector<Point> > contours;
    set<int> index_arr;
    Mat markers;
    int size;
    Mat dst;
    get_markers(src, markers, s1, c, alpha, beta, s, size, 1);

    vector<Vec3b> colors;
    
    for (size_t i = 0; i < size; i++)
    {
        int b = 13 + 2*i;// theRNG().uniform(0, 255);
        int g =  13 + 3*i;//theRNG().uniform(0, 255);
        int r =  13 + 4*i;//theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    _watershed(src, markers, dst, size, index_arr);
    get_small_segments(dst, colors, index_arr, contours);
    
    Mat dst1;
    for (int i = 0; i < contours.size(); ++i)
    {
        src_copy = dst.clone();
        if(contourArea(contours[i]) > 250)
        {
            Mat markers(dst.rows, dst.cols, CV_8UC1, Scalar(0));
            printf("%s %f\n","Area:", contourArea(contours[i]) );
            Rect box =  boundingRect(contours[i]);
            
            /*Point p1,p2;
            p1.x = box.tl().x - 5;
            p1.y = box.tl().y - 5;
            p2.x = box.br().x + 5;
            p2.y = box.br().y + 5;
            box.tl() = p1;
            box.br() = p2;*/
            Mat seg = src(box);
            

            rectangle( src_copy, box.tl(), box.br(), Scalar(0,0,255), 1, 8, 0 );
            imshow("argva", seg);
            imshow("aqw", src_copy);

            waitKey(0);

            get_markers(seg, markers, s1, c, alpha, beta, s, size, 0);
            _watershed(seg, markers, dst1, size, index_arr);
            

        }
    }
   


    printf("%d\n",contours.size() );


    
    waitKey(0);

    return 0;
    
}


/*
apply sobel and circle also on small segments generated from watershed
then match all the three*/