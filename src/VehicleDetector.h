#pragma once
#ifndef VEHICLEDETECTOR_H
#define VEHICLEDETECTOR_H

#include <opencv.hpp>

using namespace std;
using namespace cv;


// VehicleDetection struct
struct VehicleDetection{
	Rect boundingBox;
	float conf;
	Mat vehicleImage;

	VehicleDetection(){}
	VehicleDetection(Rect _boundingBox, Mat _vehicleImage, float _conf):
		boundingBox(_boundingBox),vehicleImage(_vehicleImage), conf(_conf){}
};

// VehicleDetector ascending sort predicate
struct vehicle_sort_pred{
	bool operator()(const VehicleDetection &left,const VehicleDetection &right){
		return left.conf > right.conf;
	}
};

//VehicleTrainData struct
struct VehicleTrainData{
	vector<Mat> trainingImages;
	Mat classLabels;
	int orient;

	VehicleTrainData(){}
	VehicleTrainData(vector<Mat> _trainingImages, Mat _classLabels, int _orient = 0):
		trainingImages(_trainingImages), classLabels(_classLabels), orient(_orient){}
	VehicleTrainData &operator=(const VehicleTrainData & vtd)
	{
		if(this == &vtd)
			return *this;

		this->trainingImages = vtd.trainingImages;
		this->classLabels = vtd.classLabels;
		this->orient = vtd.orient;

		return *this;
	}
};

//VTrainData struct
struct VTrainData{
	int orient;
	vector<string> posFilenames;
	vector<string> negFilenames;

	VTrainData(){};
	VTrainData(int& _orient, vector<string>& _posFiles, vector<string>& _negFiles )
		:orient(_orient), posFilenames(_posFiles), negFilenames(_negFiles){};

	void write(FileStorage& fs) const
	{
		fs << "{" << "Orientation" << orient;
		fs << "PositiveFiles" << "[";
		for(int i = 0; i < posFilenames.size(); i++)
		{
			fs << posFilenames[i];
		}
		fs << "]";
		fs << "NegativeFiles" << "[";
		for(int i = 0; i < negFilenames.size(); i++)
		{
			fs << negFilenames[i];
		}
		fs << "]";
		fs << "}";
	}
	void read(const FileNode& node)
	{
		vector<string> pf;
		vector<string> nf;
		// Read orientation
		orient = (int)node["Orientation"];
		// Read Positive Filenames
		FileNode n = node["PositiveFiles"];
		if(n.type() != FileNode::SEQ)
		{
			cout << "Error: PositiveFiles is not a sequence" << endl;
			cin.get();
			exit(0);
		}
		FileNodeIterator it = n.begin(), it_end = n.end();
		for(; it != it_end; ++it)
			pf.push_back((string)*it);
		//Read Negative Filenames
		n = node["NegativeFiles"];
		if(n.type() != FileNode::SEQ)
		{
			cout << "Error: NegativeFiles is not a sequence" << endl;
			cin.get();
			exit(0);
		}
		it = n.begin(), it_end = n.end();
		for(; it != it_end; ++it)
			nf.push_back((string)*it);

		posFilenames = pf;
		negFilenames = nf;
	}
};

// VehicleDetector Class
class VehicleDetector
{

private:
	SVM* orientedSVMs[4]; // array of 4 SVM pointers
	SVMParams svm_params;
	const int k_fold;

	Ptr<FeatureDetector> cornerDetector;

	HOGDescriptor* hog;
	static const Size win_size;
	static const Size block_size;
	static const Size block_stride;
	static const Size cell_size;
	static const int no_bins;

	/* Methods */
	void computeHOG(Mat &image, Mat &hogFeature);
	void nms(vector<VehicleDetection> &vehicleDetections, float overlapThresh);
public:
	/* Constructor */
	VehicleDetector();

	/* Methods */
	void detect(const Mat& image, vector<VehicleDetection>& vehicleDetections,
		float overlapThreshold = 0.5f, float densityThreshold = 0.0025f, bool threaded = false);

	void train(VehicleTrainData (&trainingData)[4]);

	void load(const char* filename);
	void save(const char* filename);

};


#endif