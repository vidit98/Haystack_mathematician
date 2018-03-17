#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

static const int DIST_SHIFT = 16;
static const int INIT_DIST0 = (INT_MAX >> 2);
#define  CV_FLT_TO_FIX(x,n)  cvRound((x)*(1<<(n)))
#define GRADIENT(x,y)  abs((x - y)*(1<<12))
namespace trans
{
static void
initTopBottom( Mat& temp, int border )
{
    Size size = temp.size();
    for( int i = 0; i < border; i++ )
    {
        int* ttop = temp.ptr<int>(i);
        int* tbottom = temp.ptr<int>(size.height - i - 1);

        for( int j = 0; j < size.width; j++ )
        {
            ttop[j] = INIT_DIST0;
            tbottom[j] = INIT_DIST0;
        }
    }
}

static void distanceTransform_3x3( const Mat& _src, const Mat& _gray, Mat& _temp, Mat& _dist, const float* metrics, int alpha )
{
    const int BORDER = 1;
    int i, j;
    const int HV_DIST = CV_FLT_TO_FIX( metrics[0], DIST_SHIFT );
    const int DIAG_DIST = CV_FLT_TO_FIX( metrics[1], DIST_SHIFT );
    const float scale = 1.f/(1 << DIST_SHIFT);

    const uchar* src = _src.ptr();
    const uchar* gray = _gray.ptr();
    int* temp = _temp.ptr<int>();
    float* dist = _dist.ptr<float>();
    int srcstep = (int)(_src.step/sizeof(src[0]));
    int step = (int)(_temp.step/sizeof(temp[0]));
    int dststep = (int)(_dist.step/sizeof(dist[0]));
    Size size = _src.size();

    initTopBottom( _temp, BORDER );

    // forward pass
    for( i = 0; i < size.height; i++ )
    {
        const uchar* s = src + i*srcstep;
        const uchar* g = gray + i*srcstep;
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;

        for( j = 0; j < BORDER; j++ )
            tmp[-j-1] = tmp[size.width + j] = INIT_DIST0;

        for( j = 0; j < size.width; j++ )
        {
            if( !s[j] )
                tmp[j] = 0;
            else
            {   int a = -step - 1;
                int t0 = tmp[j-step-1] + DIAG_DIST;
                int t = tmp[j-step] + HV_DIST;
                if( t0 > t ) {t0 = t ; a = -step;}
                t = tmp[j-step+1] + DIAG_DIST ;
                if( t0 > t ) {t0 = t; a = 1-step;}
                t = tmp[j-1] + HV_DIST;
                if( t0 > t ){t0 = t;a = -1;}
                tmp[j] = t0 + alpha*GRADIENT(g[j],g[j + a]);
            }
        }
    
    }

    // backward pass
    for( i = size.height - 1; i >= 0; i--)
    {
        float* d = (float*)(dist + i*dststep);
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;
        const uchar* g = gray + i*srcstep;
        for( j = size.width - 1; j >= 0; j-- )
        {
            int t0 = tmp[j];
            int a = 0;
            if( t0 > HV_DIST )
            {
                int t = tmp[j+step+1] + DIAG_DIST;
                if( t0 > t ) {t0 = t ; a = step + 1;}
                t = tmp[j+step] + HV_DIST;
                if( t0 > t ){ t0 = t; a = step;}
                t = tmp[j+step-1] + DIAG_DIST;
                if( t0 > t ){ t0 = t; a = step -1;}
                t = tmp[j+1] + HV_DIST ;
                if( t0 > t ){ t0 = t; a = 1;}
                tmp[j] = t0 + alpha*GRADIENT(g[j],g[j + a]);
            }
            d[j] = (float)(tmp[j] * scale);
        }
    }
}

static void distanceTransform_5x5( const Mat& _src, const Mat& _gray, Mat& _temp, Mat& _tempg, Mat& _dist, const float* metrics, int alpha)
{
    const int BORDER = 2;
    int i, j;
    const int HV_DIST = CV_FLT_TO_FIX( metrics[0], DIST_SHIFT );
    const int DIAG_DIST = CV_FLT_TO_FIX( metrics[1], DIST_SHIFT );
    const int LONG_DIST = CV_FLT_TO_FIX( metrics[2], DIST_SHIFT );
    const float scale = 1.f/(1 << DIST_SHIFT);

    const uchar* src = _src.ptr();
    const uchar* gray = _gray.ptr();
    int* temp = _temp.ptr<int>();
    int* tempg = _tempg.ptr<int>();
    float* dist = _dist.ptr<float>();
    int srcstep = (int)(_src.step/sizeof(src[0]));
    int graystep = (int)(_gray.step/sizeof(gray[0]));
    int step = (int)(_temp.step/sizeof(temp[0]));
    int stepg = (int)(_tempg.step/sizeof(tempg[0]));
    int dststep = (int)(_dist.step/sizeof(dist[0]));
    Size size = _src.size();
   

    initTopBottom( _temp, BORDER );
    int a = 0;
    int c = 0;
    // forward pass
    
    for( i = 0; i < size.height; i++ )
    {
        const uchar* s = src + i*srcstep;
        const uchar* g = gray + i*graystep;
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;
        int* tmpg = (int*)(tempg + (i+BORDER)*stepg) + BORDER;
        tmpg[0] = 1;
        for( j = 0; j < BORDER; j++)
        {
            tmp[-j-1] = tmp[size.width + j] = INIT_DIST0;
             
            tmpg[-j-1] = tmpg[size.width + j] = 0;
            
        }
        

        for( j = 0; j < size.width; j++ )
        {
            if( !s[j] )
            {    
                tmp[j] = 0;
                tmpg[j] = 0;
            }
            else
            {
                a = -step*2-1;
                int t0 = tmp[j-step*2-1] + LONG_DIST ;
                int t = tmp[j-step*2+1] + LONG_DIST ;
                if( t0 > t ){t0 = t; a = -graystep*2+1;c = -stepg*2+1;}
                t = tmp[j-step-2] + LONG_DIST ;
                if( t0 > t ){ t0 = t ; a = -graystep -2;c = -stepg*-2;}
                t = tmp[j-step-1] + DIAG_DIST ;
                if( t0 > t ){t0 = t; a = -graystep -1;c = -stepg-1;}
                t = tmp[j-step] + HV_DIST;
                if( t0 > t ){ t0 = t ; a = -graystep;c = -stepg;}
                t = tmp[j-step+1] + DIAG_DIST;
                if( t0 > t ){ t0 = t; a = -graystep+1;c = -stepg+1;}
                t = tmp[j-step+2] + LONG_DIST;
                if( t0 > t ){ t0 = t; a = -graystep+2;c = -stepg+2;}
                t = tmp[j-1] + HV_DIST;
                if( t0 > t ){t0 = t; a = -1;c = -1;}
                tmp[j] = t0;//temp[j+a] + alpha*GRADIENT(g[j] , g[j + a]);
                tmpg[j] = tmpg[j+c] + GRADIENT(g[j] , g[j + a]);
            }
        }
    }
    int b = a;
    // backward pass
    for( i = size.height - 1; i >= 0; i-- )
    {
        float* d = (float*)(dist + i*dststep);
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;
        int* tmpg = (int*)(tempg + (i+BORDER)*stepg) + BORDER;
        const uchar* g = gray + i*graystep;
        for( j = size.width - 1; j >= 0; j-- )
        {
            int t0 = tmp[j];
           
            if( t0 > HV_DIST )
            {
                int t = tmp[j+step*2+1] + LONG_DIST;

                if( t0 > t ){t0 = t; a = graystep*2 + 1;c = stepg*2 + 1;}
                t = tmp[j+step*2-1] + LONG_DIST ;
                if( t0 > t ){t0 = t;a = graystep*2 -1;c = stepg*2 - 1;}
                t = tmp[j+step+2] + LONG_DIST ;
                if( t0 > t ){ t0 = t; a = graystep + 2;c = stepg + 2;}
                t = tmp[j+step+1] + DIAG_DIST;
                if( t0 > t ) {t0 = t; a = graystep + 1;c = stepg + 1;}
                t = tmp[j+step] + HV_DIST ;
                if( t0 > t ){ t0 = t; a = graystep;c = stepg;} 
                t = tmp[j+step-1] + DIAG_DIST;
                if( t0 > t ){ t0 = t; a = graystep -1;c = stepg - 1;}
                t = tmp[j+step-2] + LONG_DIST ;
                if( t0 > t ) {t0 = t; a = graystep-2;c = stepg - 2;}
                t = tmp[j+1] + HV_DIST ;
                if( t0 > t ){ t0 = t; a = 1; c = 1;}
                tmp[j] = t0;//temp[j+a]+ alpha*GRADIENT(g[j] , g[j+a]);
                if(b != a);
                    tmpg[j] = tmpg[j+c] + GRADIENT(g[j] , g[j + a]);
               
            }
            if (tmpg[j]*scale > 255)
            {
                //d[j] = (float)((t0 * scale + 255));
            }
            else
            {
                d[j] = (float)((t0 + alpha*tmpg[j])* scale);
            }
            
        }
    }
}


static void getDistanceTransformMask( int maskType, float *metrics )
{
    CV_Assert( metrics != 0 );

    switch (maskType)
    {
    case 30:
        metrics[0] = 1.0f;
        metrics[1] = 1.0f;
        break;

    case 31:
        metrics[0] = 1.0f;
        metrics[1] = 2.0f;
        break;

    case 32:
        metrics[0] = 0.955f;
        metrics[1] = 1.3693f;
        break;

    case 50:
        metrics[0] = 1.0f;
        metrics[1] = 1.0f;
        metrics[2] = 2.0f;
        break;

    case 51:
        metrics[0] = 1.0f;
        metrics[1] = 2.0f;
        metrics[2] = 3.0f;
        break;

    case 52:
        metrics[0] = 1.0f;
        metrics[1] = 1.4f;
        metrics[2] = 2.1969f;
        break;
    default:
        CV_Error(CV_StsBadArg, "Unknown metric type");
    }
}


void distanceTransform( InputArray _src, InputArray _gray, OutputArray _dst,
                            int distType, int maskSize , int alpha)
{
   
    Mat gray = _gray.getMat();
    Mat src = _src.getMat(), labels;
    
    CV_Assert( src.type() == CV_8UC1);

    _dst.create( src.size(), CV_32F);
    Mat dst = _dst.getMat();

    float _mask[5] = {0};
    if( maskSize != CV_DIST_MASK_3 && maskSize != CV_DIST_MASK_5 && maskSize != CV_DIST_MASK_PRECISE )
        CV_Error( CV_StsBadSize, "Mask size should be 3 or 5 or 0 (precise)" );

    CV_Assert( distType == CV_DIST_C || distType == CV_DIST_L1 || distType == CV_DIST_L2 );

    getDistanceTransformMask( (distType == CV_DIST_C ? 0 :
        distType == CV_DIST_L1 ? 1 : 2) + maskSize*10, _mask );

    Size size = src.size();

    int border = maskSize == CV_DIST_MASK_3 ? 1 : 2;
    Mat temp( size.height + border*2, size.width + border*2, CV_32SC1 );
    Mat tempg( size.height + border*2, size.width + border*2, CV_32SC1 );
    if( maskSize == CV_DIST_MASK_3 )
    {
        distanceTransform_3x3(src, gray, temp, dst, _mask, alpha);
    }
    else
    {
        distanceTransform_5x5(src, gray, temp,tempg, dst, _mask,alpha);
    }

}


}
// int main()
// {
//     return 0;
// }