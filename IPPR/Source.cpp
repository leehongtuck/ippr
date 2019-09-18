#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace cv;
using namespace dnn;
using namespace saliency;

void compute(Mat frame);
std::vector<Mat> segment(Mat src, Mat rgb);
Mat processSobel(Mat img);
Mat KMeans(Mat src, int clusterCount);
void findContours(Mat, Mat);
void findVehicles(Mat);

//dnn functions
void classifyImage(Mat frame);
std::vector<String> getOutputsNames(const Net& net);
void postprocess(Mat& frame, const std::vector<Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

int main() {
	VideoCapture cap("./ippr.mp4");
	std::cout << "is opened: " << cap.isOpened() << std::endl;

	while (true)
	{
		Mat frame;
		for (int i = 0; i < 10; i++)
			cap >> frame;

		if (frame.empty())
			break;

		//classifyImage(frame);
		compute(frame);

		char c = (char)waitKey(10);
		if (c == 27)
			break;

		//waitKey();
		//break;
	}
	cap.release();
	destroyAllWindows();
}

void compute(Mat frame) {
	Rect roi(0, frame.rows * 0.4, frame.cols, frame.rows * 0.5);
	Mat toProcess = (frame.clone())(roi);
	cvtColor(toProcess, toProcess, COLOR_RGB2GRAY);
	equalizeHist(toProcess, toProcess);
	medianBlur(toProcess, toProcess, 3);

	Ptr<StaticSaliencySpectralResidual> salSR = StaticSaliencySpectralResidual::create();
	//Ptr<StaticSaliencyFineGrained> salFG = StaticSaliencyFineGrained::create();
	Mat mapSR, mapFG;
	salSR->computeSaliency(toProcess, mapSR);
	//salFG->computeSaliency(toProcess, mapFG);
	mapSR.convertTo(mapSR, CV_8U, 255);
	//mapFG.convertTo(mapFG, CV_8U, 255);

	cvtColor(mapSR, mapSR, COLOR_GRAY2BGR);
	//cvtColor(mapFG, mapFG, COLOR_GRAY2BGR);
	//Mat added;
	//addWeighted(mapSR, 0.7, mapFG, 0.3, 0.0, added);

	Mat kMeansMap = KMeans(mapSR, 3);
	//imshow("kmeans", kMeansMap);
	std::vector<Mat> croppedImages = segment(kMeansMap, frame(roi));
	//std::vector<Mat> croppedVehicles;

	//for (Mat img : croppedImages) {
	//	classifyImage(img);
	//}

	imshow("processed", frame);
}

Mat KMeans(Mat src, int clusterCount) {
	Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * src.rows, z) = src.at<Vec3b>(y, x)[z];

	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat dst = Mat::zeros(src.size(), src.type());
	Vec2i pointVal = { 0, 0 };

	//Get color with highest intensity
	for (int y = 0; y < centers.rows; y++) {
		int sum = 0;
		for (int x = 0; x < centers.cols; x++) {
			sum += centers.at<float>(y, x);
		}
		if (sum / 3 > pointVal[1]) {
			pointVal[0] = y;
			pointVal[1] = sum / 3;
		}
	}

	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{

			int cluster_idx = labels.at<int>(y + x * src.rows, 0);
			//dst.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			//dst.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			//dst.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
			if (cluster_idx == pointVal[0]) {
				dst.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
				dst.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
				dst.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
			}
		}

	cvtColor(dst, dst, COLOR_BGR2GRAY);
	return dst;
}

Mat processSobel(Mat img) {
	Mat src_gray;
	Mat grad;
	int ksize = 1;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	GaussianBlur(img, src_gray, Size(7, 7), 0, 0, BORDER_DEFAULT);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	//Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	abs_grad_y.convertTo(grad, CV_8U);

	return abs_grad_y;
}

std::vector<Mat> segment(Mat src, Mat ori) {

	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	std::vector<Mat> croppedImg;

	threshold(src, src, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//
	int dilation_size = 3;
	dilate(src, src,
		getStructuringElement(MORPH_RECT,
			Size(dilation_size * 2 + 1, dilation_size * 2 + 1),
			Point(dilation_size))
	);

	//int erosion_size = 1;
	//erode(src, src,
	//	getStructuringElement(MORPH_CROSS,
	//		Size(erosion_size * 2 + 1, erosion_size * 2 + 1),
	//		Point(erosion_size))
	//);
	imshow("closing", src);

	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	// Original image clone
	RNG rng(12345);

	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		Rect r = boundingRect(contours.at(i));
		//std::cout << "width: " << r.width << " height: " << r.height << "\n";

		if (r.width < ori.cols * 0.05 || r.height < ori.rows * 0.1) {
			continue;
		}

		//if (r.width > ori.cols * 0.4) {
		//	findVehicles(rgb(r));
		//}
		//else {
		//	rectangle(rgb, r, color);
		//}
		croppedImg.push_back(ori(r));
		//drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
	}
	//imshow("Contours", drawing);
	return croppedImg;
}

void findVehicles(Mat src) {
	Mat process = src.clone();
	cvtColor(process, process, COLOR_BGR2GRAY);

	equalizeHist(process, process);
	medianBlur(process, process, 5);
	Canny(process, process, mean(process)[0] * 0.66, mean(process)[0] * 1.33);

	//int erosion_size = 1;
	//erode(process, process,
	//	getStructuringElement(MORPH_ELLIPSE,
	//		Size(erosion_size * 2 + 1, 1),
	//		Point(erosion_size, 0))
	//);

	int dilation_size = 10;
	dilate(process, process,
		getStructuringElement(MORPH_ELLIPSE,
			Size(3, dilation_size * 2 + 1),
			Point(1, dilation_size))
	);

	imshow("canny", process);
	//threshold(process, process, 0, 255, THRESH_BINARY | THRESH_OTSU);

	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	findContours(process, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(process.size(), CV_8UC3);
	// Original image clone
	RNG rng(12345);

	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		Rect r = boundingRect(contours.at(i));
		//std::cout << "width: " << r.width << " height: " << r.height << "\n";
		double areaRatio = contourArea(contours[i]) / r.area();

		if (r.width > process.cols * 0.3 || r.height < process.rows * 0.45) {
			continue;
		}

		rectangle(src, r, color, 5);
		//drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
	}

	//imshow("contours", drawing);
}

//dnn variables
std::vector<std::string> classes;

void classifyImage(Mat frame) {
	// Initialize the parameters
	float confThreshold = 0.5; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;        // Width of network's input image
	int inpHeight = 416;       // Height of network's input image

	// Load names of classes
	std::string classesFile = "coco.names";
	std::ifstream ifs(classesFile.c_str());
	std::string line;

	while (getline(ifs, line)) classes.push_back(line);

	// Give the configuration and weight files for the model
	String modelConfiguration = "yolov3.cfg";
	String modelWeights = "yolov3.weights";

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open the image file
	std::string str = "Test.jpg";
	//std::ifstream ifile(str);
	//if (!ifile) throw("error");
	str.replace(str.end() - 4, str.end(), "_yolo_out.jpg");
	std::string outputFile = str;

	// Create a 4D blob from a frame.
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	//Set the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	std::vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// Remove the bounding boxes with low confidence
	postprocess(frame, outs);


	// Write the frame with the detection boxes
	//imshow("ori frame", frame);
	//Mat detectedFrame;
	//frame.convertTo(detectedFrame, CV_8U);
	//imshow("obj detection", detectedFrame);
	//imwrite(outputFile, detectedFrame);
}

std::vector<String> getOutputsNames(const Net& net)
{
	static std::vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void postprocess(Mat& frame, const std::vector<Mat>& outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;
	float confThreshold = 0.5;
	float nmsThreshold = 0.4;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

	//Get the label for the class name and its confidence
	std::string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}