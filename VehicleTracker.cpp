#include "VehicleTracker.h"

VehicleTracker::VehicleTracker(VehicleDetection det, Scalar _color, int v_no, bool _active)
{
	noFrameNotAllocated = 0;
	currDetection = det;
	color = _color;
	vehicleNo= v_no;
	active = _active;
	activity = 0;

	particleFilter = ParticleFilter((Mat_<float>(1,2)<< (det.boundingBox.x + (det.boundingBox.width/2)),
		(det.boundingBox.y + (det.boundingBox.height/2))),100);

	Mat transitionMatrix = (Mat_<float>(4,4) << 1,0,0,0, 0,1,0,0, 1,0,1,0, 0,1,0,1);
	particleFilter.setTransitionMatrix(transitionMatrix);

	Mat processNoise = (Mat_<float>(1,4) << 0.5,0.5,3.0,3.0);
	particleFilter.setProcessNoise(processNoise);

	Mat measurementNoise = (Mat_<float>(1,2) << 0.0001, 0.0001);
	particleFilter.setMeasurementNoise(measurementNoise);
}

void VehicleTracker::predict(void)
{
	particleFilter.predict();
}

bool VehicleTracker::update(VehicleDetection detection)
{
	// Assign vehicleDetection to tracker
	setCurrDetection(detection);
	// Reset numberOfFramesNotAllocated
	noFrameNotAllocated = 0;
	// Increment activity
	activity++;

	int x_c = detection.boundingBox.x+(detection.boundingBox.width/2);
	int y_c = detection.boundingBox.y+(detection.boundingBox.height/2);
	// Particle filter correction/resampling step
	Mat measurement = (Mat_<float>(1,2) << x_c, y_c);
	particleFilter.correct(measurement);

	// If the tracker is in-active and has been
	// has been assigned n detections then become active
	if(!active && activity > 1)
	{
		active = true;
		return true;
	}
	//Returing active change status - kinda hacky!
	return false;
}

double VehicleTracker::score(VehicleDetection detection)
{
	double score;

	// Compute histogram intersection score
	Mat hsv_CurrDetection, hsv_Detection;
	cvtColor(currDetection.vehicleImage,hsv_CurrDetection,CV_BGR2HSV);
	cvtColor(detection.vehicleImage, hsv_Detection, CV_BGR2HSV);

	int histSize[] = { 50, 32 };
	float h_ranges[] = { 0 , 256 };
	float s_ranges[] = { 0, 180 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0 , 1 };

	MatND hist_CurrDetection, hist_Detection;
	calcHist(&hsv_CurrDetection,1,channels,Mat(),hist_CurrDetection,2,histSize,ranges,true,false);
	normalize(hist_CurrDetection, hist_CurrDetection, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_Detection,1,channels,Mat(),hist_Detection,2,histSize,ranges,true,false);
	normalize(hist_Detection, hist_Detection, 0, 1, NORM_MINMAX, -1, Mat());

	double histInterScore = compareHist(hist_CurrDetection,hist_Detection,CV_COMP_CORREL);

	// Comupte distanceScore
	int vdCenterX = detection.boundingBox.x + (detection.boundingBox.width /2);
	int vdCenterY = detection.boundingBox.y + (detection.boundingBox.height /2);
	Mat measurement = (Mat_<double>(1,2)<< vdCenterX, vdCenterY);
	double distanceScore = particleFilter.score(measurement);

	score = histInterScore + distanceScore;
	return score;
}

void VehicleTracker::setCurrDetection(VehicleDetection det)
{
	prevDetection = currDetection;
	currDetection = det;
}

VehicleDetection VehicleTracker::getCurrDetection(void)
{
	return currDetection;
}

void VehicleTracker::drawParticles(Mat *image)
{
	for(int i = 0; i < particleFilter.particles.size(); i++)
	{
		circle(*image,Point(particleFilter.particles[i].state.at<float>(0,0),
			particleFilter.particles[i].state.at<float>(0,1)),1,color);
	}
}

void VehicleTracker::setVehicleNo(int num)
{
	vehicleNo = num;
}