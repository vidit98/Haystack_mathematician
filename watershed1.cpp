#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cstdio>
#include <iostream>  
#include <set>
#include "Transform.cpp"
#include "bg_subtract.cpp"
#include "sobel.cpp"

using namespace cv;
using namespace std;

int alpha = 2 , beta = 155 , c = 7 , s = 1 , s1 = 18, cou=0,cc=0;
vector<Mat> segm;
vector<Mat> segm2;
vector<Point > segp;
vector<vector<Point> > seg_contour;
Mat src,finalimg,ptr;
Mat temp, t, imgg;
vector<Point > main_contour;

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
    //imshow("q" , bw1);
    imwrite("bn1.jpg",bw1);
    bilateralFilter(bw1, bw, 2*alpha + 1, beta, 50 );
    
    //imshow("Binary Image1", bw);
    imwrite("a.jpg" , bw);
    // threshold(bw, bw2, 40, 255, CV_THRESH_BINARY_INV| CV_THRESH_OTSU);
    adaptiveThreshold(bw, bw2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 2*s1 + 1, c);
    ////imshow("Binary Image", bw2);
    imwrite("a1.jpg" , bw2);

    // Perform the distance transform algorithm
    Mat dist(img.rows, img.cols, CV_16UC1,Scalar(0));
    //distanceTransform(bw2,dist, CV_DIST_L2, 5);
    trans::distanceTransform(bw2,bw1,dist, CV_DIST_L2, 5,s);
    // Normalize the distance image for range = {0.0, 1.0}
    //so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    dist.convertTo(dist, CV_32F);
    
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
       //if (contourArea(contours[i]) > (img.rows*img.cols/8000))
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
    
    //imshow("Marker", src_copy);
   
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
    //imshow("Final Result", dst);
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

void process_small(Mat dst, vector<Vec3b> colors,  set<int> index_arr, int area, int app_area)
{
    int radius = int(sqrt(app_area/3.14))-5 ;
    cout<<radius<<endl;
    //int radius = 10;
    imwrite("dst.jpg", dst);
    Mat src_copy = dst.clone();
    imwrite("srccopy.jpg", src_copy);
    
    if(area>app_area)
    {
    vector<vector <Point> > contours;
   
    get_small_segments(dst, colors, index_arr, contours);
        if(contours.size()>=1)
        {
            for (int i=0; i < contours.size(); ++i)
            {
                if(!cou)
                {
                    
                    if(contourArea(contours[i]) > app_area )
                    {
                        seg_contour.push_back(contours[i]);
                        Mat markers(dst.rows, dst.cols, CV_8UC1, Scalar(0));
                        printf("%s %f %d \n","Area:", contourArea(contours[i]),cou );
                        Rect box = boundingRect(contours[i]);
                        segp.push_back(box.tl());
                        Mat seg = src(box);
                        Mat seg2 = t(box);
                        
                        segm.push_back(seg);
                        segm2.push_back(seg2);
                        
                        rectangle( src_copy, box.tl(), box.br(), Scalar(0,0,255), 1, 8, 0 );
                        //imshow("argva", seg);
                        imshow("aqw", src_copy);

                        waitKey(0);
            
                    } 
                    else
                    {
                        
                        int sumi=0,sumj=0;
                        for(int j=0;j<contours[i].size();j++)
                        {
                            sumi+=contours[i][j].y;
                            sumj+=contours[i][j].x;
                        }
                        sumi/=contours[i].size();
                        
                        sumj/=contours[i].size();
                        
                        //drawContours(ddd, contours, i,Scalar(colors[i][0],colors[i][1],colors[i][2]), -1);
                        
                        //if((int)(pointPolygonTest(seg_contour[0], Point(sumj,sumi), false))<0)
                        circle(finalimg, Point(sumj,sumi), 3, CV_RGB(255,0,0), -1);
                        
                    }   
                    Rect r = boundingRect(Mat(contours[i]));
                        imgg = temp.clone();
                        rectangle( imgg,Point(r.tl().x, r.tl().y), Point(r.br().x, r.br().y), Scalar(0,0,255), 1, 8, 0 );
                        
                        
                        Mat img_ROI = t(r);
                        Mat lapc;
                            lapc = conv_to_laplace(img_ROI);
                        cout<<"inside else !cou\n";
                        imshow("lapalce", lapc);
                        waitKey(0);
                        vector<Point> wrt; 
                        for(int m=0;m<contours[i].size();m++){
                            Point pp ;
                            pp.x = contours[i][m].x - r.tl().x;
                            pp.y = contours[i][m].y - r.tl().y;
                            wrt.push_back(pp); 
                        }                 
                        
                        vector<Point> sob_point = apply_sobel(lapc, wrt, radius);
                       
                        for(int k = 0;k<sob_point.size();k++){
                            if((int)(pointPolygonTest(main_contour,Point(r.tl().x + sob_point[k].x, r.tl().y + sob_point[k].y), false))>0)
                            circle(ptr, Point(r.tl().x + sob_point[k].x, r.tl().y + sob_point[k].y), 2, CV_RGB(255,0,0), -1);
                            circle(imgg, Point(r.tl().x + sob_point[k].x, r.tl().y + sob_point[k].y), 2, CV_RGB(255,50*k,0), -1);
                            //printf("%s %d %d\n","k", sob_point[k].x, sob_point[k].y );
                        }
                        imwrite("sobelresult.jpg",ptr);
                        imshow("sobel", imgg);
                        waitKey(0);

                }

                else
                {
                    bool ans1 = (matchShapes(contours[i],seg_contour[0],CV_CONTOURS_MATCH_I1,0) > 1e-2);
                    if((contourArea(contours[i]) > app_area) && ans1)
                    {
                        imgg = temp.clone();
                        rectangle( imgg,Point(segp[0].x, segp[0].y), Point(segm[0].cols+segp[0].x, segm[0].rows + segp[0].y), Scalar(0,0,255), 1, 8, 0 );
                        imshow("sobel1", imgg);
                        waitKey(0);
                        seg_contour.push_back(contours[i]);
                        Mat markers(dst.rows, dst.cols, CV_8UC1, Scalar(0));
                        printf("%s %f %d \n","Area:", contourArea(contours[i]),cou );
                        Rect box =  boundingRect(contours[i]);
                        segp.push_back(Point(box.tl().x+segp[0].x,box.tl().y+segp[0].y));
                        Mat seg = segm[0](box);
                        Mat seg2 = segm2[0](box);
                        
                        segm.push_back(seg);
                        segm2.push_back(seg2);
                        rectangle( src_copy, box.tl(), box.br(), Scalar(0,0,255), 1, 8, 0 );
                        //imshow("argva", seg);
                        imshow("aqw", src_copy);

                        waitKey(0);
            
                    } 
                    else
                    {
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
                        //Mat ddd(segm2[0].rows, segm2[0].cols, CV_8UC3, Scalar(0, 0, 0));
                        //drawContours(ddd, contours, i,Scalar(colors[i][0],colors[i][1],colors[i][2]), -1);
                        //imwrite("finalimg1.jpg", ddd);
                        
                        if((int)(pointPolygonTest(seg_contour[0], Point(sumj,sumi), false))>0){
                            circle(finalimg, Point(sumj,sumi), 3, CV_RGB(255,0,0), -1);
                            /*printf("less than 400\n");
                            imgg = temp.clone();
                            Rect r = boundingRect(Mat(contours[i]));
                            rectangle( imgg,Point(r.tl().x+segp[0].x, r.tl().y + segp[0].y), Point(r.br().x+segp[0].x, r.br().y + segp[0].y), Scalar(0,0,255), 1, 8, 0 );
                            imshow("sobel", imgg);
                            waitKey(0);
                            Mat img_ROI = segm2[0](r);
                            cout<<img_ROI.rows<<" "<<img_ROI.cols<<endl;
                            Mat lapc;
                            lapc = conv_to_laplace(img_ROI);  
                            imshow("laplace", lapc);
                            waitKey(0);
                            vector<Point> wrt;
                            for(int m=0;m<contours[i].size();m++){
                                Point pp ;
                                pp.x = contours[i][m].x - r.tl().x;
                                pp.y = contours[i][m].y - r.tl().y;
                                wrt.push_back(pp); 
                            }

                            vector<Point> sob_point = apply_sobel(lapc, wrt, radius);
                            printf("%s %d\n","k", sob_point.size());
                            for(int k = 0;k<sob_point.size();k++){
                                circle(finalimg, Point(r.tl().x + segp[0].x + sob_point[k].x, r.tl().y + segp[0].y + sob_point[k].y), 2, CV_RGB(255,0,0), -1);
                                circle(imgg, Point(r.tl().x + segp[0].x + sob_point[k].x, r.tl().y + segp[0].y + sob_point[k].y), 2, CV_RGB(255,20*k,0), -1);
                                printf("%s %d %d\n","k",r.tl().x + sob_point[k].x, r.tl().y + sob_point[k].y);
                                //circle(segm2[0], Point(sob_point[k].x,sob_point[k].y), 2, Scalar(255), -1);
                            }
                            imshow("sobel", imgg);
                            waitKey(0);*/

                        }

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
    temp = imread(argv[1]); 
    
    
    //Mat temp1(400, 400, CV_8UC3, Scalar(0,0,0));
    // resize(temp, temp1, Size(temp1.rows, temp1.cols));
    // imshow("win",temp1);

    
    src = temp.clone();
    ptr = temp.clone(); 
    finalimg=temp.clone();
    
    if (!temp.data)   
        return -1; 
 
    Mat yohoo; 
    
    //imshow("Black Background Image", src);
    
    bg(yohoo,temp,main_contour);
    cvtColor(yohoo, t, CV_BGR2GRAY);
    sharp_image(yohoo , src);
    imwrite("sharp.jpg", src);
    //imshow("a" , src); 
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
    int app_area = calc_radius(dst, colors);
    cout<<"the calculated area is "<<app_area<<endl;
    waitKey(10);
    process_small(dst,colors,index_arr,temp.rows*temp.cols, app_area+500);
    
    int i = 0;
    
    while(!segm.empty())
    {
        Mat dst1, dst2;
        cou++;
        imwrite("seg.jpg", segm[0]);
        index_arr.clear();
        get_markers(segm[0], markers, s1, c, alpha, beta, s, size, 0);
        _watershed(segm[0], markers, dst1, size, index_arr, colors);
        imwrite("win.jpg",dst1);
        waitKey(0);
        process_small(dst1,colors,index_arr,contourArea(seg_contour[0]), app_area);
        segm.erase(segm.begin());
        segm2.erase(segm2.begin());
        seg_contour.erase(seg_contour.begin());
        segp.erase(segp.begin());
    }
   

    imwrite("finalimg.jpg",finalimg);
    imwrite("ptr.jpg",ptr);
    waitKey(0);

    return 0;
    
}


/*
apply sobel and circle also on small segments generated from watershed
then match all the three*/