#include "utils.h"

void keep_terminal_open(void)
{
	//Keep terminal open
	cout << "Press Enter to continue ... " << flush;
	cin.ignore( numeric_limits<streamsize>::max(), '\n');
}

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */

int64 GetTimeMs64()
{
#ifdef WIN32
	/* Windows */
	FILETIME ft;
	LARGE_INTEGER li;

	/* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
	 * to a LARGE_INTEGER structure. */
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;

	uint64 ret = li.QuadPart;
	ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
	ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

	return ret;
#else
	/* Linux */
	struct timeval tv;

	gettimeofday(&tv, NULL);

	uint64 ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	ret /= 1000;

	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	ret += (tv.tv_sec * 1000);

	return ret;
#endif
}

void rotateImage(Mat source, Mat *dest, double angle)
{	
	Point2f src_center(source.cols/2.0f, source.rows/2.0f);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    warpAffine(source, *dest, rot_mat, source.size());
}

void integralImage(Mat &src, Mat &dst)
{
	dst = cv::Mat::zeros(src.size(),CV_32FC1);
	for( int ix = 0; ix < dst.rows; ix++)
	{
		for(int iy = 0; iy < dst.cols; iy++)
		{
			if(ix == 0 && iy == 0){
				dst.at<int>(ix,iy) = src.at<int>(ix,iy);
			}else if(ix == 0){
				dst.at<int>(ix,iy) = src.at<int>(ix,iy) + dst.at<int>(ix,iy - 1);
			}else if(iy == 0){
				dst.at<int>(ix,iy) = src.at<int>(ix,iy) + dst.at<int>(ix - 1,iy);
			}else{
				dst.at<int>(ix,iy) = src.at<int>(ix,iy) + dst.at<int>(ix - 1,iy)
					+ dst.at<int>(ix,iy - 1) - dst.at<int>(ix - 1,iy - 1);
			}
		}
	}
}