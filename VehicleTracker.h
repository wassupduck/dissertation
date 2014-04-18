#pragma once
#ifndef VEHICLETRACKER_H
#define VEHICLETRACKER_H

#include <opencv.hpp>
#include "VehicleDetector.h"
#include "ParticleFilter.h"

using namespace std;
using namespace cv;

class VehicleTracker
{

private:

	VehicleDetection prevDetection;
	VehicleDetection currDetection;
	int activity;

	ParticleFilter particleFilter;

	void setCurrDetection(VehicleDetection det);

public:	

	int noFrameNotAllocated;
	bool active;
	
	Scalar color;
	int vehicleNo;

	// Constructors
	VehicleTracker(void);
	VehicleTracker(VehicleDetection det, Scalar _color, int v_no = -1, bool _active = false);
	// Methods
	VehicleDetection getCurrDetection(void);
	void setVehicleNo(int num);

	void predict(void);
	bool update(VehicleDetection detection);
	double score(VehicleDetection detection);

	//Debugging functions
	void drawParticles(Mat *image);
};

#endif