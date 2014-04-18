#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

#include<opencv.hpp>

#include "Utils.h"
#include "ImageLoader.h"
#include "VehicleDetector.h"
#include "VehicleTracker.h"
#include <ctime>

using namespace std;
using namespace cv;

void loadTrainingData(char* filname, VehicleTrainData (&vehicleTrainData)[4]);
void trackVehicles(char* filename, char* model = NULL);
void trainVehicleDetector(char* filename, char* outputFile);
void help(void);

RNG rngen(static_cast<unsigned int>(time(NULL)));

void help(void)
{
	cout << "Usage is\nvts.exe <filename> - Track vehicles in filename" << endl;
	cout << "vts.exe -t <filename> <output model> -Train the vehicle Detector" << endl;
	cout << "vts.exe -m <model> <filename> - Track vehicles in filename using (vehicle)model" << endl;
	cin.get();
	exit(0);
}

int main(int argc, char* argv[])
{
	if(argc > 4){ // Check enough parameters have been passed
		help();
	}else{ // if we get enough parameters
		if(argc == 2)
		{
			// vts.exe <filename>
			char* filename = argv[1];
			trackVehicles(filename);
		}else if (argc == 4)
		{
			string argv1 = argv[1];
			if(argv1 == "-t")
			{
				// vts.exe -t <filename> <output model>
				char* filename = argv[2];
				char* out_model = argv[3];
				trainVehicleDetector(filename,out_model);
			}else if(argv1 == "-m")
			{
				// vts.exe -m <model> <filename>
				char* filename = argv[3];
				char* model = argv[2];
				trackVehicles(filename, model);
			}else{
				help();
			}
		}else{
			help();
		}
	}
	cin.get();
	return 0;
}

/*
 * Main tracking function
 */
void trackVehicles(char* filename, char* model)
{
	/* ----- Capture Initialisation ----- */
	cv::VideoCapture capture(filename);
	if(!capture.isOpened()){
		cout << "Video could not be opened" << endl;
		exit(0);
	}

	/* ----- Vehicle Detector Initalisation ----- */
	// Create new Vehicle Detector
	VehicleDetector vehicleDetector;
	// Load Vehicle Detector model
	if(model == NULL){
		//load default model
		vehicleDetector.load(".\\example_model.xml");
	}else vehicleDetector.load(model);

	/* ----- Main Tracking Loop ----- */
	Mat frame;
	vector<VehicleDetection> vehicleDetections;
	namedWindow("Detections");
	bool firstFrame = true;
	vector<VehicleTracker> trackers;
	int maxTrackNo;
	vector<bool> tracker_assigned, detection_assigned;

	// Parameters
	// Detector params
	const float overlapThresh = 0.45f;
	const float densityThresh = 0.006f; //0.005f
	// Tracking params
	const double scoreThresh = 1.1;

	for(;;)
	{
		// Read frame
		capture >> frame;
        if (frame.empty())
			break;

		// Detect vehicles in frame
		vehicleDetector.detect(frame,vehicleDetections,
								overlapThresh,densityThresh,false);

		// Create detection image for display
		Mat detectionImg;
		frame.copyTo(detectionImg);

		if(firstFrame)
		{
			// First frame
			// Create 'active' tracker for each detector response
			for(int i = 0; i < vehicleDetections.size(); i++)
			{
				trackers.push_back(VehicleTracker(vehicleDetections[i],
									Scalar(rngen.uniform(0, 255), rngen.uniform(0,255), rngen.uniform(0,255))
									,i,true));
			}
			maxTrackNo = trackers.size()-1;
			// All trackers assigned detection
			for(int _t =0; _t < trackers.size(); _t++) tracker_assigned.push_back(true);
			firstFrame = false;
		}else{
			Mat scoreMatrix = Mat_<double>(trackers.size(),vehicleDetections.size());
			// N'th frame

			
			// Particle filter predict step
			for(int tp=0; tp < trackers.size(); tp++)
			{
				trackers[tp].predict();
				//Draw Particles
				//if(trackers[tp].active)
				//	trackers[tp].drawParticles(&detectionImg);
			}
			// Link trackers and vehicleDetections

			// Compute score matrix
			for(int d = 0; d < vehicleDetections.size(); d++)
			{
				for(int t =0; t < trackers.size(); t++)
				{
					double score = trackers[t].score(vehicleDetections[d]);

					if(score > scoreThresh)
						scoreMatrix.at<double>(t,d) = score;
					else
						scoreMatrix.at<double>(t,d) = -1;
				}
			}

			tracker_assigned.clear();
			detection_assigned.clear();
			for(int _t =0; _t < trackers.size(); _t++)
				tracker_assigned.push_back(false);
			for(int _d =0; _d < vehicleDetections.size(); _d++)
				detection_assigned.push_back(false);

			int track, detect;
			double largest = 0;
			bool cont = true;
			bool active_pass = true;

			// Assign vehicleDetections to existing 'active' tracks then to 'in-active' tracks
			// Largest scoring track/detection pair assigned first
			while(cont){
				largest = 0;
				for(int i = 0; i < scoreMatrix.cols; i++)
				{
					for(int j = 0; j < scoreMatrix.rows; j++)
					{
						if(tracker_assigned[j] == false && detection_assigned[i] == false 
							&& scoreMatrix.at<double>(j,i) >= 0 && scoreMatrix.at<double>(j,i) > largest)
						{
							//If active pass then check tracker is active
							if(active_pass){
								if(trackers[j].active){
									largest = scoreMatrix.at<double>(j,i);
									track = j;
									detect = i;
								}
							}else{
								largest = scoreMatrix.at<double>(j,i);
								track = j;
								detect = i;
							}
						}
					}
				}
				if(largest != 0){
					// Update tracker
					// Assign vehicleDetection to tracker
					bool actChange = trackers[track].update(vehicleDetections[detect]);
					if(actChange) trackers[track].setVehicleNo(++maxTrackNo);
					// Mark track and vehicleDetection as assigned
					// i.e. remove row and column from score matrix
					tracker_assigned[track] = true;
					detection_assigned[detect] = true;
				}else if(active_pass){
					active_pass = false;
				}else cont = false;
			}

			// Spawn new 'in-active' trackers for each detector response
			// not allocated to a tracker
			for(int s=0; s< detection_assigned.size(); s++)
			{
				if(!detection_assigned[s])
				{
					trackers.push_back(VehicleTracker(vehicleDetections[s],
						Scalar(rngen.uniform(0, 255), rngen.uniform(0,255), rngen.uniform(0,255)),
						-1,false));
					tracker_assigned.push_back(true);
				}
			}
			// Reap any trackers that have not recieved a
			// detector response for 3 frames;
			for(int r=0; r < tracker_assigned.size(); r++)
			{
				if(!tracker_assigned[r])
				{
					trackers[r].noFrameNotAllocated++;
					if(trackers[r].noFrameNotAllocated > 2)
					{
						trackers.erase(trackers.begin()+r);
						tracker_assigned.erase(tracker_assigned.begin()+r);
					}
				}
			}
		}

		// Display tracker current detections
		for(int d = 0; d < trackers.size(); d++)
		{
			if(trackers[d].active && tracker_assigned[d]){
				rectangle(detectionImg,trackers[d].getCurrDetection().boundingBox,trackers[d].color,2);
				putText(detectionImg,"V:"+
					static_cast<ostringstream*>(&(ostringstream() << trackers[d].vehicleNo))->str()
					,trackers[d].getCurrDetection().boundingBox.tl(),
					1,1,trackers[d].color);
			}
		}
		imshow("Detections", detectionImg);
		cv::waitKey(1);

	}
}

// Functions to write/read VTrainData from FileStorage
void write(FileStorage& fs, const string&, const VTrainData& x)
{
	x.write(fs);
}

void read(const FileNode& node, VTrainData& x, const VTrainData& defualt_value = VTrainData())
{
	if(node.empty())
		x = defualt_value;
	else{
		x.read(node);
	}
}

void loadTrainingData(char* filename, VehicleTrainData (&vehicleTrainData)[4])
{
	try{
		if((sizeof vehicleTrainData / sizeof VehicleTrainData) != 4)
		{
			throw "VehicleTrainData Array must have size of 4";
		}

		FileStorage fs(filename,FileStorage::READ);

		FileNode n = fs["VTrainDatas"];
		if(n.type() != FileNode::SEQ)
		{
			throw "VTrainDatas  is not a sequence";
		}

		vector<VTrainData> vTrainingFiles;
		//Load VTrainData into memory
		FileNodeIterator it = n.begin(), it_end = n.end();
		for(int i =0; it != it_end; ++it, i++)
		{
			vTrainingFiles.push_back(VTrainData());
			(*it) >> vTrainingFiles[i];
		}

		if(vTrainingFiles.size() != 4){
			throw "Four VTrainData structures are required to train vehicleDetector";
		}

		vector<Mat> pos_training_imgs;
		vector<Mat> neg_training_imgs;
		int orient;

		for(int j = 0; j < vTrainingFiles.size(); j++)
		{
			//Get orientation
			orient = vTrainingFiles[j].orient;
			/* ----- Load tarining images ----- */
			//Load positive image files
			for(int p = 0; p < vTrainingFiles[j].posFilenames.size(); p++)
			{
				pos_training_imgs.push_back(imread(vTrainingFiles[j].posFilenames[p],CV_LOAD_IMAGE_COLOR));
			}
			//Load negative image files
			for(int n = 0; n < vTrainingFiles[j].negFilenames.size(); n++)
			{
				neg_training_imgs.push_back(imread(vTrainingFiles[j].negFilenames[n],CV_LOAD_IMAGE_COLOR));
			}
			vector<cv::Mat> training_imgs = pos_training_imgs;
			training_imgs.insert(training_imgs.end(), neg_training_imgs.begin(), neg_training_imgs.end());

			/* ----- Build response matrix for training data ----- */
			cv::Mat training_responses = cv::Mat_<float>(1, training_imgs.size());
			training_responses.colRange(0, pos_training_imgs.size()) = 1.0f;
			training_responses.colRange(pos_training_imgs.size(), training_responses.cols) = -1.0f;

			vehicleTrainData[j] = VehicleTrainData(training_imgs,training_responses,orient);

			pos_training_imgs.clear();
			neg_training_imgs.clear();
		}

		fs.release();

	}catch(const exception& ex)
	{
		cout << "Error: " << ex.what() << endl;
		cin.get(); //keep_terminal_open();
		exit(0);
	}
}

void trainVehicleDetector(char* filename, char* outputFile)
{
	// Load training data from *filename*
	VehicleTrainData vehicleTrainData[4];
	loadTrainingData(filename,vehicleTrainData);

	// Train a vehicle Detector
	VehicleDetector vehicleDetector;
	vehicleDetector.train(vehicleTrainData);

	// Save vehicleDetector SVM 
	vehicleDetector.save(outputFile);

	cout << "Training Completed!" << endl;
	cout << "Vehicle Detector model saved to: " << outputFile << endl;

}