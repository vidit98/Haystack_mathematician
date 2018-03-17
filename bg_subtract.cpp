#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cstdio>
#include <iostream>
#include <set>

using namespace cv;
using namespace std;

void bg(Mat& img, Mat src, vector<Point >& mainc)
{
	int max=0,pos=0;
	img=src.clone();
	//Mat img=imread("bg2.jpg",0);
	//Mat src=imread("test5.jpg",1);
	Mat dst(img.rows,img.cols,CV_8UC1,Scalar(0));
	//namedWindow("erode", WINDOW_AUTOSIZE);
	//createTrackbar("h", "erode", &e, 7);
	Mat imgLaplacian, sharp1, imgResult;
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1);

    filter2D(src, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp1, CV_32F);
    imgResult = sharp1 - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

	int alpha = 2 , beta = 155 , s = 1 , s1 = 18 , c=7;
	Mat bw,bw1,bw2;
    Mat src_copy = src.clone();
    cvtColor(src, bw1, CV_BGR2GRAY);
       
    medianBlur(bw1,bw1,3);
    bilateralFilter(bw1, bw, 2*alpha + 1, beta, 50 );
    
    adaptiveThreshold(bw,  bw2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 2*s1 + 1, c);	
			Mat img1;
			Mat element = getStructuringElement( MORPH_ELLIPSE,Size( 3,3 ),Point( 1,1 ) );
			dilate(bw2,bw2,element);
			dilate(bw2,bw2,element);
			dilate(bw2,bw2,element);
			dilate(bw2,img1,element);
			//erode( bw2, bw2, element );
			//erode( bw2, img1, element );
			//imshow("r",img1);
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
    		findContours(img1, contours,hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    		for(int j=0;j<contours.size();j++)
    		{
    			if(contourArea(contours[j])>max )
    			{
    				max=contourArea(contours[j]);
    				pos=j;
    			}
    		}
    		cout<<contourArea(contours[hierarchy[pos][2]])<<" "<<max<<endl;
    		drawContours(dst, contours, pos,Scalar(255), -1);
    		mainc=contours[pos];
    		erode( dst, dst, element );
    		erode( dst, dst, element );
    		erode( dst, dst, element );
    		erode( dst, dst, element );
    		erode( dst, dst, element );
    		erode( dst, dst, element );
    		erode( dst, dst, element );
    		erode( dst, dst, element );
    		//drawContours(dst, contours, -1,Scalar(255), 1);
    		//imshow("yo",dst);
    		for(int i=0;i<img.rows;i++)
    			for(int j=0;j<img.cols;j++)
    			{
    				if(dst.at<uchar>(i,j)==0)
    				{
    					img.at<Vec3b>(i,j)={220,220,220};
    				}
    			}
			imshow("win",img);
			imwrite("bgsub.jpg",img);
			waitKey(0);
}