#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char** argv)
{
    // Load the image
    Mat src = imread(argv[1]);
    
    // Check if everything was fine
    if (!src.data)
        return -1;
    // Show source image
    imshow("Source Image", src);
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    for( int x = 0; x < src.rows; x++ ) {
      for( int y = 0; y < src.cols; y++ ) {
          if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
            src.at<Vec3b>(x, y)[0] = 0;
            src.at<Vec3b>(x, y)[1] = 0;
            src.at<Vec3b>(x, y)[2] = 0;
          }
        }
    }
    // Show output image
    imshow("Black Background Image", src);
    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    Mat sharp = src; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    imshow( "Laplace Filtered Image", imgLaplacian );
    imshow( "New Sharped Image", imgResult );
    src = imgResult; // copy back
    // Create binary image from source image
    Mat bw,bw1;
    int alpha = 2 , beta = 155 , acc = 50 , minl = 20;
    namedWindow("Object Detection",WINDOW_AUTOSIZE);
    createTrackbar("alpha","Object Detection", &alpha, 50);
    createTrackbar("beta","Object Detection", &beta, 180);

    while(1)
    {
        Mat src_copy = imread(argv[1]);

        cvtColor(src, bw1, CV_BGR2GRAY);

        medianBlur(bw1,bw,3);
        imshow("q" , bw1);
        imwrite("bn1.jpg",bw1);
        bilateralFilter(bw1, bw, 2*alpha + 1, beta, 50 );
        
        
        imwrite("a.jpg" , bw);
        threshold(bw, bw, 40, 255, CV_THRESH_BINARY_INV| CV_THRESH_OTSU);
        //adaptiveThreshold(bw,  bw, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 31, 8);
        imshow("Binary Image", bw);
        imwrite("a1.jpg" , bw);
        // Perform the distance transform algorithm
        Mat dist;
        distanceTransform(bw, dist, CV_DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        normalize(dist, dist, 0, 1., NORM_MINMAX);
        imshow("Distance Transform Image", dist);
        imwrite("Distancet.jpg", dist);
        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
        // Dilate a bit the dist image
        Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
        dilate(dist, dist, kernel1);
        imshow("Peaks", dist);
        imwrite("Peaks.jpg", dist);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        Mat dist_8u;
        dist.convertTo(dist_8u, CV_8U);
        // Find total markers
        vector<vector<Point> > contours;
        findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        Mat markers = Mat::zeros(dist.size(), CV_32SC1);
        // Draw the foreground markers
        for (size_t i = 0; i < contours.size(); i++)
            {
               // if (contourArea(contours[i]) < 200)
                {
                    drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
                    drawContours(src_copy, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

                }
            }

        // Draw the background marker
        circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
        // circle(markers, Point(dist.rows-5,5), 3, CV_RGB(255,255,255), -1);
        // circle(markers, Point(dist.rows-5,dist.cols -5), 3, CV_RGB(255,255,255), -1);
        // circle(src_copy, Point(dist.rows-5,5), 3, CV_RGB(0,255,255), -1);
        // circle(src_copy, Point(dist.rows-5,dist.cols -5), 3, CV_RGB(0,255,255), -1);

        circle(src_copy, Point(5,5), 3, CV_RGB(0,255,255), -1);
        imshow("Markers", src_copy);
        imwrite("Markers.jpg", src_copy);
        // Perform the watershed algorithm
        watershed(src, markers);
        Mat mark = Mat::zeros(markers.size(), CV_8UC1);
        markers.convertTo(mark, CV_8UC1);
        bitwise_not(mark, mark);
        //imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                      // image looks like at that point
        // Generate random colors
        vector<Vec3b> colors;
        for (size_t i = 0; i < contours.size(); i++)
        {
            int b = 10 + 13*i;// theRNG().uniform(0, 255);
            int g =  10 + 13*i;//theRNG().uniform(0, 255);
            int r =  10 + 13*i;//theRNG().uniform(0, 255);
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
                if (index > 0 && index <= static_cast<int>(contours.size()))
                    dst.at<Vec3b>(i,j) = colors[index-1];
                else
                    dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
        }
        // Visualize the final image
        imshow("Final Result", dst);
        imwrite("final.jpg", dst);
        waitKey(10);
    }
    return 0;
    
}