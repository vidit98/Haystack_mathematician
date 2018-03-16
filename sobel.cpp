#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


int is_valid(int x, int y, int rows, int cols){

    if( x>=0 && x<rows && y>=0 && y<cols)
        return 1;
    else 
        return 0;
}

int calc_radius(Mat dst, vector<Vec3b> colors){

	Mat frame_thresh;
    vector<Point>area;

    for(int i = 0; i<colors.size(); i++){
   
        inRange(dst, colors[i], colors[i], frame_thresh);
           
        vector<vector<Point> > contours2;
        findContours(frame_thresh, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        cout<<contours2.size()<<endl;
            
        for(int j=0; j<contours2.size(); j++){
        	if(contourArea(contours2[j])>150){
	        	Rect r = boundingRect(Mat(contours2[j]));
	        	int w = r.br().x - r.tl().x;
	        	int h = r.br().y - r.tl().y;
	        	float ratio = ((float)w)/h;
	        	bool a1 = ratio > 0.7;
	        	bool a2 = ratio < 1.3;

	        	if(a1 && a2){

	        		int ar = (int)(contourArea(contours2[j]));
	        		ar = (ar/100+1)*100;
	                    //cout<<ar<<endl;
	        		int flag = 0;
	            	for(int i=0;i<area.size();i++){
	            		if(area[i].x == ar){
	            			area[i].y++;
	            			flag = 1;
	            			break;
	            		}
	            	}
	            	if(!flag){
	            		area.push_back(Point(ar, 1));
	            	}    
	                
	        	}
	        }	
        }

    }

    int max_ar = 0;
    int ind = 0;
    for(int i=0;i<area.size();i++){
        cout<<area[i].x<<" "<<area[i].y<<endl;
        if(area[i].y>=max_ar){
            max_ar = area[i].y;
            ind = i;
        }
    }

    //cout<<"the radius is "<<area[ind].x<<endl;

    return area[ind].x;
}




vector<Point> apply_sobel(Mat img, vector<Point> cont, int radius){

	

	int rows = img.rows;
    int cols = img.cols;
    int max_radius = radius + 3;
    int min_radius = radius;

    vector<Point> v;
    vector<int> r; 

    for(int i = 0; i<img.rows; i++){
        for(int j = 0; j<img.cols; j++){

        	if(int(pointPolygonTest(cont, Point(j,i), false)) > 0){
        		//cout<<"enter\n";
	            int intensity = img.at<uchar>(i, j);
	            int k = min_radius;
	            int k_opt = 0, ans;
	            
	            int gradient = 0;
	            for(;k<=max_radius;k++){
	                if(img.at<uchar>(i-k, j)-img.at<uchar>(i-k-1, j) > gradient && img.at<uchar>(i-k, j) - intensity >=10){
	                    k_opt = k;
	                    gradient = img.at<uchar>(i-k, j)-img.at<uchar>(i-k-1, j); 
	                }
	                if(img.at<uchar>(i+k, j)-img.at<uchar>(i+k+1, j) > gradient && img.at<uchar>(i+k, j) - intensity >=10){
	                    k_opt = k;
	                    gradient = img.at<uchar>(i+k, j)-img.at<uchar>(i+k+1, j); 
	                }
	                if(img.at<uchar>(i, j+k)-img.at<uchar>(i, j+k+1) > gradient && img.at<uchar>(i, j+k) - intensity >=10){
	                    k_opt = k;
	                    gradient = img.at<uchar>(i, j+k+1)-img.at<uchar>(i, j+k+1); 
	                }
	                if(img.at<uchar>(i, j-k)-img.at<uchar>(i, j-k-1) > gradient && img.at<uchar>(i, j-k) - intensity >=10){
	                    k_opt = k;
	                    gradient = img.at<uchar>(i, j-k)-img.at<uchar>(i, j-k-1); 
	                }

	            }
	            
	            
	            //k = max(k_opt[0], max(k_opt[1], max(k_opt[2], k_opt[3])));
	            k = k_opt;
	            if(k>=min_radius && k<=max_radius)
	                k = k - 1;
	            else 
	                continue;

	            int total = 0;
	            int max_length = 0;
	            int flag = 0;
	            int max_intensity = 0;
	            
	            //printf("%s\n", "check1");
	            vector<int> binary;
	            vector<int> count;
	            vector<int> avg_int;

	            for(int l=0;l<360;l++)
	            {
	                 
	                int x = j + (int)(k*cos(l*3.14/180));
	                int y = i + (int)(k*sin(l*3.14/180));
	                int intensity_avg = 0;
	                if(flag == 0 )
	                {
	                    if(count.size() > 0)
	                        count[count.size() - 1] += 1;
	                    if(is_valid(y, x, rows, cols) && img.at<uchar>(y,x) - intensity >=10)
	                    {
	                        flag = 1;
	                        total = 1;
	                        ans = img.at<uchar>(y,x);
	                        binary.push_back(1);
	                        count.push_back(1);
	                        intensity_avg+=ans;

	                    }
	                }
	                else{

	                    if(is_valid(y, x, rows, cols))
	                    {
	                        if(abs(img.at<uchar>(y,x) - ans)<=20 && (img.at<uchar>(y,x) >= 10))
	                        {
	                            count[count.size() - 1] +=1;
	                            intensity_avg += img.at<uchar>(y,x);
	                            total++;
	                        }
	                        else
	                        {
	                            avg_int.push_back(intensity_avg/total);
	                            binary.push_back(0);
	                            avg_int.push_back(0);
	                            count.push_back(1);
	                            if(total>max_length)
	                                max_length = total;
	                            total = 0;
	                            flag = 0;
	                            intensity_avg = 0;
	                         } 

	                        ans = img.at<uchar>(y,x);

	                    }
	                }
	                
	                
	            }
	           

	            if(count.size() > 0)
	            {
	               
	                max_length = 0;
	                int length = count[0];
	                //printf("%s %d\n", "length", length);

	                for (int q = 0; q < count.size()-1; ++q)
	                {
	                    if (binary[q] == 0 && count[q] <= 2 && abs(avg_int[q-1] - avg_int[q+1]) < 20)
	                    {
	                        length += count[q+1]; 
	                    }
	                    else if(binary[q] == 0 && count[q] > 1)
	                    {
	                        if (length > max_length)
	                        {
	                            max_length = length;
	                            
	                        }
	                        length = count[q+1];
	                    }
	                }

	                if (length > max_length)
	                    max_length = length;
	                            
	                        
	                // printf("%s %d\n","length", max_length );
	                // printf("%d %d\n",i,j );
	            }
	            
	            if(max_length > 50)
	            {
	            	printf("%s %d %d\n","Cricle detected", i, j);
	                v.push_back(Point(j,i));
	                r.push_back(k);
	                circle( img, Point(j, i), 1, Scalar(255), -1, 8, 0 );
	            }

            }   
            
        }
    }
   
    imwrite("laplace.jpg", img);
    return v;
}

Mat conv_to_laplace(Mat img){

	GaussianBlur( img, img, Size(3, 3), 2, 2 );
    Mat dst;
    Canny(img, dst, 20, 70, 3);

    dilate(dst, dst, Mat(), Point(-1, -1), 2);
    //imshow("dialtion for laplacian", dst);
    //waitKey(0);
    vector<vector<Point> > contours;
    findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    int ind = 0;
    float ar = 0;


    for(int i=0;i<contours.size();i++){
        if(contourArea(contours[i])>ar){
            ind = i;
            ar = contourArea(contours[i]);
        }
    }


    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); 
    Mat imgLaplacian;
    Mat sharp = img;
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    img.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC1);

    if(contours.size() > 0){

	    for(int i=0; i<imgLaplacian.rows; i++){
	        for(int j=0; j<imgLaplacian.cols; j++){

	                if(pointPolygonTest(contours[ind], Point(j,i), false)<0)
	                    imgLaplacian.at<uchar>(i, j) = 0;

	        }
	    }
	}

    return imgLaplacian;
}

