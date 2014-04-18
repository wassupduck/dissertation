#include "VehicleDetector.h"
#include "Utils.h"
#include <math.h>

VehicleDetector::VehicleDetector():k_fold(10)
{
	//Initalise SVM's
	orientedSVMs[0] = new SVM();
	orientedSVMs[1] = new SVM();	orientedSVMs[2] = new SVM();
	orientedSVMs[3] = new SVM();

	//Initalise SVM parameters
	svm_params.svm_type = cv::SVM::C_SVC;
	svm_params.kernel_type = cv::SVM::RBF;
	svm_params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	//Initalise HOG descriptor
	hog = new HOGDescriptor(win_size,block_size,block_stride,cell_size,no_bins,1,-1,cv::HOGDescriptor::L2Hys,0.2,true,cv::HOGDescriptor::DEFAULT_NLEVELS);

	//Initalise FAST detector
	cornerDetector = new FastFeatureDetector(10,true);

}

const Size VehicleDetector::win_size = Size(64,64);
const Size VehicleDetector::block_size = Size(32,32);
const Size VehicleDetector::block_stride = Size(32,32);
const Size VehicleDetector::cell_size = Size(16,16);
const int VehicleDetector::no_bins = 21;

void VehicleDetector::train(VehicleTrainData (&trainingData)[4])
{
	try{
		if( (sizeof trainingData / sizeof VehicleTrainData) != 4)
			throw "VehicleDetector trainingData must contain 4 VehicleTrainData structure";

		Mat training_data_mat(0,hog->getDescriptorSize(),CV_32FC1);
		Mat labels_mat(0,1,CV_32FC1);
		Mat hogFeature(1,hog->getDescriptorSize(),CV_32FC1);

		//Train each SVM orientation
		for(int i = 0; i < 4; i++)
		{
			for(int j = 0; j< trainingData[i].trainingImages.size(); j++)
			{
				computeHOG(trainingData[i].trainingImages[j],hogFeature);
				training_data_mat.push_back(hogFeature);
				labels_mat.push_back(trainingData[i].classLabels.at<float>(0,j));
			}
			switch(trainingData[i].orient)
			{
			case 0:
				orientedSVMs[0]->train_auto(training_data_mat,labels_mat,Mat(),Mat(),svm_params,k_fold);
				break;
			case 45:
				orientedSVMs[1]->train_auto(training_data_mat,labels_mat,Mat(),Mat(),svm_params,k_fold);
				break;
			case 90:
				orientedSVMs[2]->train_auto(training_data_mat,labels_mat,Mat(),Mat(),svm_params,k_fold);
				break;
			case 135:
				orientedSVMs[3]->train_auto(training_data_mat,labels_mat,Mat(),Mat(),svm_params,k_fold);
				break;
			default:
				throw "Orientation angle not expected";
			}
			//clear data
			training_data_mat.resize(0);
			labels_mat.resize(0);
		}
	}catch(const exception& ex)
	{
		cerr << "Error: " << ex.what() << endl;
		cin.get(); //keep_terminal_open();
		exit(0);
	}
}

void VehicleDetector::computeHOG(Mat &image, Mat &hogFeature)
{
	Mat *im = &Mat();
	vector<float> desc;

	if(image.rows != win_size.height || image.cols != win_size.width)
		resize(image,*im,win_size);
	else
		im = &image;

	hog->compute(*im,desc);
	for(int i =0; i < desc.size(); i++)
	{
		hogFeature.at<float>(0,i) = desc.at(i);
	}
}

void VehicleDetector::detect(const Mat& image, vector<VehicleDetection>& vehicleDetections,
		float overlapThreshold, float densityThreshold, bool threaded)
{
	int maxLevel = 1;
	int wind_width = 64;
	int wind_height = 64;
	int dx = 15;
	int dy = 15;
	int sx,sy;
	float score;

	Mat grayFrame;
	vector<KeyPoint> keypoints;

	cvtColor(image,grayFrame,CV_BGR2GRAY);
	vehicleDetections.clear();

	/* ----- ROI selection ----- */
	vector<Rect> ROI;
	cornerDetector->detect(grayFrame,keypoints);

	//Create corner image
	Mat cornerImage = Mat::zeros(grayFrame.size(),CV_32FC1);
	for(int c = 0; c < keypoints.size(); c++)
	{
		cornerImage.at<int>(keypoints[c].pt) = 1;
	}
	//Create integral image
	Mat integralImg;
	integralImage(cornerImage,integralImg);

	//Feature density estimation

	int y_across, x_down;
	for(int l = 0; l < maxLevel; l++)
	{
		y_across = floor((float)(integralImg.cols - wind_width)/dy)+1;
		x_down = floor((float)(integralImg.rows - wind_height)/dx)+1;
		int wind_area = wind_width*wind_height;
		for(int a= 0; a < y_across; a++)
		{
			sy = a*dy;
			for(int b = 0; b < x_down; b++)
			{
				sx=b*dx;
				score = (float)(integralImg.at<int>(sx+wind_height,sy+wind_width) - integralImg.at<int>(sx,sy+wind_width)
							- integralImg.at<int>(sx+wind_height,sy) + integralImg.at<int>(sx,sy))/(float)(wind_area);
						if(score > densityThreshold) ROI.push_back(cv::Rect(sy,sx,wind_width,wind_height));
			}
		}
		wind_width += (wind_width/2);
		wind_height += (wind_height/2);
	}

	//Classify ROIs
	Mat hogFeature(1,hog->getDescriptorSize(),CV_32FC1);
	float high_conf = 0.0f;
	float conf;
	for(int r = 0; r < ROI.size(); r++)
	{
		computeHOG(image(ROI[r]),hogFeature);
		//For each SVM orientation
		for(int orient =0; orient < 4; orient++)
		{
			conf = orientedSVMs[orient]->predict(hogFeature,true);
			high_conf = (conf < high_conf)?conf:high_conf;
		}
		if(high_conf < 0){ //If it is a vehicle
			vehicleDetections.push_back(VehicleDetection(ROI[r],image(ROI[r]).clone(),abs(high_conf)));
		}
		high_conf = 0.0f;
	}

	//Non maximum suppression
	nms(vehicleDetections,overlapThreshold);
}

void VehicleDetector::nms(vector<VehicleDetection> &vehicleDetections, float overlapThresh)
{
	sort(vehicleDetections.begin(),vehicleDetections.end(),vehicle_sort_pred());
	for(int v=0; v < vehicleDetections.size(); v++)
	{
		int x11 = vehicleDetections[v].boundingBox.tl().x;
		int y11 = vehicleDetections[v].boundingBox.tl().y;
		int x12 = vehicleDetections[v].boundingBox.br().x;
		int y12 = vehicleDetections[v].boundingBox.br().y;

		for(int h=v+1; h < vehicleDetections.size(); h++)
		{
			int x21 = vehicleDetections[h].boundingBox.tl().x;
			int y21 = vehicleDetections[h].boundingBox.tl().y;
			int x22 = vehicleDetections[h].boundingBox.br().x;
			int y22 = vehicleDetections[h].boundingBox.br().y;

			int si = max(0,min(x12,x22)-max(x11,x21))*max(0,min(y12,y22)-max(y11,y21));
			// ratio of combined area
			//int su = vehicleDetections[v].boundingBox.area() + vehicleDetections[h].boundingBox.area() - si;
			//float ratio = (float)si/(float)su;

			// ratio of overlap area
			float ratio = (float)si/(float)vehicleDetections[h].boundingBox.area();
			if(ratio > overlapThresh){
				vehicleDetections.erase(vehicleDetections.begin()+h);
				h--;
			}
		}
	}
}

void VehicleDetector::save(const char* filename)
{
	CvFileStorage* fs;
	fs = cvOpenFileStorage(filename,NULL,CV_STORAGE_WRITE);

	if(fs == NULL){
		cout << filename << " could not be opened" << endl;
		cin.get();
		exit(0);
	}

	orientedSVMs[0]->write(fs,"SVM_0");
	orientedSVMs[1]->write(fs,"SVM_45");
	orientedSVMs[2]->write(fs,"SVM_90");
	orientedSVMs[3]->write(fs,"SVM_135");

	cvReleaseFileStorage(&fs);
}

void VehicleDetector::load(const char* filename)
{
	CvFileStorage* fs;
	fs = cvOpenFileStorage(filename,NULL,CV_STORAGE_READ);

	if(fs == NULL){
		cout << "default model not available - please provide model" << endl;
		cin.get();
		exit(0);
	}

	CvFileNode* node;
	node = cvGetFileNodeByName(fs,NULL,"SVM_0");
	orientedSVMs[0]->read(fs,node);
	node = cvGetFileNodeByName(fs,NULL,"SVM_45");
	orientedSVMs[1]->read(fs,node);
	node = cvGetFileNodeByName(fs,NULL,"SVM_90");
	orientedSVMs[2]->read(fs,node);
	node = cvGetFileNodeByName(fs,NULL,"SVM_135");
	orientedSVMs[3]->read(fs,node);

	cvReleaseFileStorage(&fs);
}

