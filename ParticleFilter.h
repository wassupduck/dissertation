#pragma once
#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

#include <opencv.hpp>

using namespace std;
using namespace cv;

struct Particle
{
	Mat state;
	Particle(Mat _state):state(_state){}
};

class ParticleFilter
{
private:

public:
	Mat processNoise;
	Mat measurementNoise;
	Mat transitionMatrix;
	vector<Particle> particles;
	vector<double> weights;
	int N;

	ParticleFilter();
	ParticleFilter(Mat center, int numParticles = 100);

	Mat predict();
	Mat correct(Mat measurement);
	double score(Mat measurement);

	void setProcessNoise(Mat _processNoise);
	void setMeasurementNoise(Mat _measurementNoise);
	void setTransitionMatrix(Mat _transitionMatrix);
};

#endif