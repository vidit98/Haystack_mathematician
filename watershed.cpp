#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cstdio>
#include <iostream>  
#include <set>
#include "Transform.cpp"
#include "bg_subtract.cpp"

using namespace cv;
using namespace std;

int alpha = 2 , beta = 155 , c = 7 , s = 1 , s1 = 18, cou=0,cc=0;
vector<Mat> segm;
vector<Point > segp;
vector<vector<Point> > seg_contour;
Mat src,finalimg,ptr;

void sharp_image(Mat img, Mat& output)
{
    // do the laplacianfiltering as it is
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

void _watershed(Mat src, Mat markers, Mat& output, int size, set<int>& index_arr, vector<Vec3b> colors)
{
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
   
    
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
       
       // printf("%d %d\n", idx);
        inRange(dst, Scalar(colors[idx][0] ,colors[idx][1], colors[idx][2]), Scalar(colors[idx][0] + 2,colors[idx][1] + 2, colors[idx][2] + 2), bw3);

        Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
        dilate(bw3, bw3, kernel1);
        border(bw3);

        // imshow("edge", bw3);      
        // imshow("edge1", bw3);
        imwrite("edge1.jpg", bw3);
        findContours(bw3, contours_tmp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        //cout<<"b\n";
        drawContours(output, contours_tmp, 0,color, -1);
       // imwrite("output.jpg",output);
        if(!contours_tmp.empty())
        {
        //cout<<"c\n";
        contours.push_back(contours_tmp[0]);
        //cout<<"d\n";
        }
        //imshow("dst", output);
       
        
   

        
    }
}

void process_small(Mat dst, vector<Vec3b> colors,  set<int> index_arr, int area)
{
    //cout<<"hi\n";
     imwrite("dst.jpg", dst);
    Mat src_copy = dst.clone();
    //vector <Point > cont;
   // cont=seg_contour[0];
 imwrite("srccopy.jpg", src_copy);
    
    if(area>400)
    {
       // cout<<cou<<endl;
    vector<vector <Point> > contours;
   
    get_small_segments(dst, colors, index_arr, contours);
    //cout<<"a\n";
        if(contours.size()>1)
        {
            for (int i=0; i < contours.size(); ++i)
            {
                //cout<<"b\n";
                if(!cou)
                {
                   
                    if(contourArea(contours[i]) > 400 )
                    {
                        cout<<"a1\n";
                        seg_contour.push_back(contours[i]);
                        Mat markers(dst.rows, dst.cols, CV_8UC1, Scalar(0));
                        printf("%s %f %d \n","Area:", contourArea(contours[i]),cou );
                        Rect box =  boundingRect(contours[i]);
                        segp.push_back(box.tl());
                        Mat seg = src(box);
                        //Mat seg1 = ptr(box);
                        segm.push_back(seg);
                        //imwrite("part1.jpg",seg1);
                        rectangle( src_copy, box.tl(), box.br(), Scalar(0,0,255), 1, 8, 0 );
                        imshow("argva", seg);
                        imshow("aqw", src_copy);

                        waitKey(0);
            
                    } 
                    else
                    {
                        //cc++;
                       // cout<<"A\n";
                        int sumi=0,sumj=0;
                        for(int j=0;j<contours[i].size();j++)
                        {
                            sumi+=contours[i][j].y;
                            sumj+=contours[i][j].x;
                        }
                        sumi/=contours[i].size();
                        //sumi+=segp[0].y;
                        sumj/=contours[i].size();
                        //sumj+=segp[0].x;
                       // drawContours(finalimg, contours, i,Scalar(colors[i][0],colors[i][1],colors[i][2]), -1);
                         imwrite("finalimg.jpg",finalimg);
       // cout<<cc<<endl;
                         //if((int)(pointPolygonTest(seg_contour[0], Point(sumj,sumi), false))<0)
                            circle(finalimg, Point(sumj,sumi), 3, CV_RGB(255,0,0), -1);
                    }  
                }

                else
                {
                    bool ans1 = (matchShapes(contours[i],seg_contour[0],CV_CONTOURS_MATCH_I1,0) != 0);
                    //cout<<ans1<<endl;
                    if(contourArea(contours[i]) > 400 && ans1)
                    {
                        //cout<<"a2\n";
                        seg_contour.push_back(contours[i]);
                       Mat markers(dst.rows, dst.cols, CV_8UC1, Scalar(0));
                        printf("%s %f %d \n","Area:", contourArea(contours[i]),cou );
                        Rect box =  boundingRect(contours[i]);
                        segp.push_back(Point(box.tl().x+segp[0].x,box.tl().y+segp[0].y));
                        Mat seg = src(box);
                        //Mat seg1 = ptr(box);
                        segm.push_back(seg);
                        //imwrite("part2.jpg",seg1);
                        rectangle( src_copy, box.tl(), box.br(), Scalar(0,0,255), 1, 8, 0 );
                        imshow("argva", seg);
                        imshow("aqw", src_copy);

                        waitKey(0);
            
                    } 
                    else
                    {
                        //cc++;
                       // cout<<"A\n";
                        int sumi=0,sumj=0;
                        for(int j=0;j<contours[i].size();j++)
                        {
                            sumi+=contours[i][j].y;
                            sumj+=contours[i][j].x;
                        }
                        sumi/=contours[i].size();
                        sumi+=segp[0].y;
                        sumj/=contours[i].size();
                        sumj+=segp[0].x;
                        //drawContours(finalimg, contours, i,Scalar(colors[i][0],colors[i][1],colors[i][2]), -1);
                         imwrite("finalimg.jpg",finalimg);
                        // cout<<cc<<endl;
                        if((int)(pointPolygonTest(seg_contour[0], Point(sumj,sumi), false))<0)
                            circle(finalimg, Point(sumj,sumi), 3, CV_RGB(255,0,0), -1);
                    }
                }
            }
        }
    }

    /*else
    {
        //cc++;
        //cout<<"A\n";
        int sumi=0,sumj=0;
        for(int i=0;i<seg_contour[0].size();i++)
        {
            sumi+=seg_contour[0][i].y;
            sumj+=seg_contour[0][i].x;
        }
        sumi/=seg_contour[0].size();
        sumj/=seg_contour[0].size();
        //cout<<cc<<endl;
         circle(finalimg, Point(sumj,sumi), 3, CV_RGB(255,0,0), -1);
    }*/
}

      
int main(int, char** argv)
{
    // Load the image
    Mat temp = imread(argv[1]); 
    //Mat temp(400, 400, CV_8UC3, Scalar(0,0,0));
   // resize(temp1, temp, Size(temp.rows, temp.cols));
    //imwrite("test5.jpg",temp);  
     src = temp.clone();
     ptr = temp.clone(); 
     finalimg=temp.clone();
    // Check if everything was fine
    if (!temp.data)   
        return -1;
 
    Mat yohoo; 
    
    imshow("Black Background Image", src);
    // Create a kernel that we will use for accuting/sharpening our image4
    bg(yohoo,temp);
    
   /* Mat imgLaplacian;
   */
    sharp_image(yohoo , src);
    imwrite("sharp.jpg", src);
    imshow("a" , src); 
    waitKey(0);
    //Create binary image from source image
    vector<vector<Point> > contours;
    Mat src_copy = temp.clone();
    
    set<int> index_arr;
    Mat markers;
    int size;
    Mat dst;
    get_markers(src, markers, s1, c, alpha, beta, s, size, 1);
    
    vector<Vec3b> colors;
    for (size_t i = 0; i < size; i++)
    {
        int b = 13 + 4*i;// theRNG().uniform(0, 255);
        int g =  13 + 3*i;//theRNG().uniform(0, 255);
        int r =  13 + 2*i;//theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    _watershed(src, markers, dst, size, index_arr,colors);
    //get_small_segments(dst, colors, index_arr, contours);
    process_small(dst,colors,index_arr,5000);
    
    int i = 0;
    
    while(!segm.empty())
    {
        Mat dst1;
        cou++;
        imwrite("seg.jpg", segm[0]);
        index_arr.clear();
         get_markers(segm[0], markers, s1, c, alpha, beta, s, size, 0);
         //cout<<"hi\n";
        _watershed(segm[0], markers, dst1, size, index_arr,colors);
        //cout<<"a\n";
        imwrite("win.jpg",dst1);
        waitKey(0);
        process_small(dst1,colors,index_arr,contourArea(seg_contour[0]));
       // cout<<"b\n";
        //cout<<segm.size()<<endl;
        segm.erase(segm.begin());
        seg_contour.erase(seg_contour.begin());
        segp.erase(segp.begin());
    }
   


   // printf("%d\n",contours.size() );


    imwrite("finalimg.jpg",finalimg);
    waitKey(0);

    return 0;
    
}


/*
apply sobel and circle also on small segments generated from watershed
then match all the three*/