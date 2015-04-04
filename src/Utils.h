#pragma once
#ifndef UTILS_H
#define UTILS_H

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

#include <opencv.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <math.h>

using namespace std;
using namespace cv;

void keep_terminal_open(void);
int64 GetTimeMs64(void);
void rotateImage(Mat source, Mat *dest, double angle);
void integralImage(Mat &src, Mat &dst);

#define drawCross( img, center, color, d)                                 \
                line( img, cv::Point( center.x - d, center.y - d ),                \
                             cv::Point( center.x + d, center.y + d ), color, 1, CV_AA, 0); \
                line( img, cv::Point( center.x + d, center.y - d ),                \
                             cv::Point( center.x - d, center.y + d ), color, 1, CV_AA, 0 )

#endif