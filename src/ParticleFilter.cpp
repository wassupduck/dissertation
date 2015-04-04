/* ----- Domain specific particle filter ----- */
#include "ParticleFilter.h"
#include <algorithm>
#include <ctime>
#include <numeric>
#define _USE_MATH_DEFINES
#include <math.h>

RNG rng(static_cast<unsigned int >(time(NULL)));

ParticleFilter::ParticleFilter(void){}

ParticleFilter::ParticleFilter(Mat center, int numParticles)
{
	//Initalise N particle each drawn independently from a 2d gaussion
	//with mean 'center'
	N = numParticles;
	for(int i = 0; i < N; i++)
	{
		Particle p((Mat_<float>(1,4) << center.at<float>(0,0) + rng.gaussian(10.0),
			center.at<float>(0,1) + rng.gaussian(10.0), rng.gaussian(3.0), rng.gaussian(3.0)));
		particles.push_back(p);
	}
}

void ParticleFilter::setMeasurementNoise(Mat _measurementNoise)
{
	measurementNoise = _measurementNoise;
}

void ParticleFilter::setProcessNoise(Mat _processNoise)
{
	processNoise = _processNoise;
}

void ParticleFilter::setTransitionMatrix(Mat _transitionMatrix)
{
	transitionMatrix = _transitionMatrix;
}

Mat ParticleFilter::predict()
{
	//Update each particles position with constant velocity model
	Mat vectorsum = Mat_<float>::zeros(1,4);
	for(int i = 0; i < N; i++)
	{
		particles[i].state = (particles[i].state*transitionMatrix)
			+(Mat_<float>(1,4) << rng.gaussian(processNoise.at<float>(0,0)),rng.gaussian(processNoise.at<float>(0,1)),
			rng.gaussian(processNoise.at<float>(0,2)),rng.gaussian(processNoise.at<float>(0,3)));
		vectorsum +=particles[i].state;
	}
	return vectorsum*(double)(1.0/N); //Prediction
}

double ParticleFilter::score(Mat measurement)
{
	//P(Y_n|X_n)
	double score = 0;
	for(int i = 0; i < particles.size(); i++)
	{
		Mat pos = (Mat_<float>(1,2) << particles[i].state.at<float>(0,0), 
			particles[i].state.at<float>(0,1));
		
		float m_x = measurement.at<double>(0,0);
		float m_y = measurement.at<double>(0,1);
		float p_x = pos.at<float>(0,0);
		float p_y = pos.at<float>(0,1);

		float sigma = 20.0;
		score +=(1.0/(sigma*sqrt(2.0*M_PI)))*exp(-(
			(((m_x-p_x)*(m_x-p_x))/(2.0*(sigma*sigma)))+(((m_y-p_y)*(m_y-p_y))/(2.0*(sigma*sigma)))));
	}
	return score;
}

Mat ParticleFilter::correct(Mat measurement)
{
	//Add some noise to measurements
	measurement += (Mat_<float>(1,2) << rng.gaussian(measurementNoise.at<float>(0,0)),
										rng.gaussian(measurementNoise.at<float>(0,1)));
	weights.clear();
	//Calculate particle importance weights
	for(int i = 0; i < N; i++)
	{
		Mat pos = (Mat_<float>(1,2) << particles[i].state.at<float>(0,0), 
			particles[i].state.at<float>(0,1));

		Mat_<float> diff = pos - measurement;
		double dist = cv::sqrt(diff.dot(diff));
		float sigma = 5.0;
		weights.push_back(exp(-(dist*dist)/(sigma*sigma)/2.0)/sqrt(2.0*M_PI*(sigma*sigma)));
	}
	//Resampling
	Mat vectorsum = Mat_<float>::zeros(1,4);
	vector<Particle> p;
	int index = (int)(rng.uniform(0.f,1.f)*N);
	double beta = 0.0f;
	double mw = *max_element(weights.begin(),weights.end());
	for(int j = 0; j < N; j++)
	{
		beta += rng.uniform(0.f,1.f) * 2.0 * mw;
		while(beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % N;
		}
		p.push_back(Particle(particles[index].state.clone()));
		vectorsum += p.back().state;
	}
	particles = p;
	return vectorsum*(double)(1.0/N); //Estimate
}