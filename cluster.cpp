#include <cstdio>
#include <iostream>  
#include <set>
#include <stdlib.h>

using namespace std;

void process_points(vector<Point> v, int& mean_x, int& mean_y, int& sd_x, int& sd_y)
{
	mean_y = 0;
	mean_x = 0;
	sd_x = 0;
	sd_y = 0;
	for (int i = 0; i < v.size(); ++i)
	{
		mean_x += v[i].x; 
		mean_y += v[i].y;
	}
	mean_y/= v.size();
	mean_x/= v.size();

	for (int i = 0; i < v.size(); ++i)
	{
		sd_y += pow(v[i].y - mean_y,2);
		sd_x += pow(v[i].x - mean_x,2);

	}
	sd_x/= v.size();
	sd_y/= v.size();

	sd_y = sqrt(sd_y);
	sd_x = sqrt(sd_x);
}