//
// Written by Andrey Leshenko, Eli Tarnarutsky and Shir Amir.
// Published under the MIT license.
//

#include <iostream>
#include <chrono>
#include <vector>

#include <opencv2/opencv.hpp>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using std::vector;
using std::cout;
using std::endl;

using cv::VideoCapture;
using cv::Mat;
using cv::Mat_;
using cv::Size;
using cv::Scalar;
using cv::Point;
using cv::RNG;
using cv::Rect;

void usage()
{
    std::cout << "Usage: camera-framerate [CAMERA_INDEX]\n"
	      << '\n'
	      << "Measure the framerate of camera number CAMERA_INDEX.\n"
	      << "If no camera index is specified, CAMERA_INDEX=0 is assumed.\n";
}

int main(int argc, char *argv[])
{
    int cameraIndex = 0;

    if (argc > 2)
    {
		usage();
		return 1;
    }

    if (argc == 2)
    {
		char *arg = argv[1];
		char *end;
		cameraIndex = std::strtol(arg, &end, 0);

		if (end - arg != std::strlen(arg))
		{
			usage();
			return 1;
		}
    }

    VideoCapture cap{cameraIndex};
    // TODO(Andrey): pass these settings as arguments
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(cv::CAP_PROP_FPS, 120);

    if (!cap.isOpened())
    {
	std::cerr << "error: couldn't capture camera number " << cameraIndex << '\n';
	return 1;
    }

    Mat currFrame;
    Mat grayscale;

    auto beginTime = high_resolution_clock::now();
	bool paused = false;
    int frameCount = 0;

	bool firstIter = true;
	int oldThresholdValue;

    int pressedKey = 0;
	int mode = 1;
	char innerMode = 'a';
	bool boarderLines = true;

	int cannyLowThreshold = 80;
	int cannyHighThreshold = 125;

	double rho = 1;
	int thetaDivider = 180;
	int houghThreshold = 50;
	int minLineLength = 50;
	int maxLineGap = 75;
	int srn = 0;
	int stn = 0;

    while (true)
    {
		if(paused != true)
		{
			cap >> currFrame;
		}

		cv::cvtColor(currFrame, grayscale, CV_BGR2GRAY);

		if (mode == 1)
		{
			cv::imshow("w", grayscale);
		}

		Mat blurred;
		cv::blur(grayscale, blurred, Size{5, 5});

		if (mode == 2)
		{
			cv::imshow("w", blurred);
		}

		Mat cannyDetection;

		cv::Canny(blurred, cannyDetection, cannyLowThreshold, cannyHighThreshold);
		cv::dilate(cannyDetection, cannyDetection, Mat());

		if (mode == 3)
		{
			cv::imshow("w", cannyDetection);
			cv::createTrackbar("Low Threshold", "w", &cannyLowThreshold, 200);
			cv::createTrackbar("High Threshold", "w", &cannyHighThreshold, 200);
		}

        // cv::dilate(cannyDetection, cannyDetection, Mat());
        // Mat element = getStructuringElement(cv::MORPH_ELLIPSE, Size(1, 1));
		// morphologyEx(cannyDetection, cannyDetection, cv::MORPH_OPEN, element);
		// morphologyEx(cannyDetection, cannyDetection, cv::MORPH_OPEN, element);

        // if(mode == 8)
        // {
        //     cv::imshow("w", cannyDetection);
        // }

		Mat cannyDetectionBlured;
		cv::blur(cannyDetection, cannyDetectionBlured, Size{8, 8});

		if(mode == 4)
		{
			cv::imshow("w", cannyDetectionBlured);
		}

		Mat drawing = Mat::zeros(cannyDetectionBlured.size(), CV_8UC3);
		Mat biggest = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
		int chosenContour = 0;

		int maxArea = 0;
		RNG rng(12345);
        int rows = cannyDetectionBlured.size[0] - 1;
		int cols = cannyDetectionBlured.size[1] - 1;

		if(boarderLines)
		{
			int range = 10;

			for(int i = range;  i < rows - range; i++)
			{
				cannyDetectionBlured.at<uchar>(i, 1) = (uchar)255;
				cannyDetectionBlured.at<uchar>(i, cols - 1) = (uchar)255;
			}

			for(int i = range;  i < cols - range; i++)
			{
				cannyDetectionBlured.at<uchar>(1, i) = (uchar)255;
				cannyDetectionBlured.at<uchar>(rows - 1, i) = (uchar)255;
			}
		}

		vector<vector<Point>> contours;
		cv::findContours(cannyDetectionBlured, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

		for(int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
			cv::drawContours(drawing, contours, i, color);

			double currArea = cv::contourArea(contours[i]);
			if (currArea > maxArea)
			{
				maxArea = currArea;
				chosenContour = i;
			}
		}

		//Mat mask = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
		// cv::drawContours(biggest, contours, chosenContour, Scalar(255), -1);

		vector<Point> newCont;

		for(Point p : contours[chosenContour])
		{
			if(p.x != 1 && p.y != 1 && p.y != rows - 1 && p.x != cols - 1)
			{
				newCont.push_back(p);
			}
		}

		contours[chosenContour] = newCont;
		cv::drawContours(biggest, contours, chosenContour, Scalar(255), -1);

		if(mode == 5)
		{
			cv::imshow("w", drawing);
		}

		if(mode == 6)
		{
			cv::imshow("w", biggest);
		}

		// if(mode == 'v')
		// {
		// 	Rect drawRect = cv::boundingRect(contours[chosenContour]);
		// 	rectangle(biggest, drawRect, Scalar(255));
		// 	cv::imshow("w", biggest);
		// }

		if(mode == 7)
		{
			cv::RotatedRect drawRect = cv::minAreaRect(contours[chosenContour]);
			cv::Point2f vertices[4];
			drawRect.points(vertices);
			for (int i = 0; i < 4; i++)
				line(biggest, vertices[i], vertices[(i+1)%4], Scalar(255));

			cv::imshow("w", biggest);
		}

		Mat newImage = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
		bitwise_and(grayscale, biggest, newImage);

		// if(mode == 7)
		// {
		// 	cv::imshow("w", newImage);
		// }

		//Mat_<CV_8U> newIm {newImage};

		//cout << (int)grayscale.at<uchar>(0, 0) << endl;
		//cout << newImage.size()<< endl;

		vector<int> pointsInsideContour;
		for(int i = 0; i < newImage.cols; i++)
		{
			for(int j = 0; j < newImage.rows; j++)
			{
				if((int)newImage.at<uchar>(i, j) > 0)
				{
					pointsInsideContour.push_back((int)newImage.at<uchar>(i, j));
				}
			}
		}

		// cout << pointsInsideContour.size() << endl;

		Mat finalImageMean;

		int sum = 0;
		int avg = 0;

		for(auto& greyVal : pointsInsideContour)
		{
			sum += greyVal;
		}

		avg = sum / pointsInsideContour.size();
		//cout << avg << endl;

		int thresholdValue = avg + 20;

		// if(firstIter)
		// {
		// 	oldThresholdValue = thresholdValue;
		// 	firstIter = false;
		// }
		// else
		// {
		// 	if(abs(oldThresholdValue - thresholdValue) > 40)
		// 	{
		// 		paused = true;
		// 	}
		// 	oldThresholdValue = thresholdValue;
		// }

		cv::threshold(grayscale, finalImageMean, thresholdValue, 255, cv::THRESH_BINARY_INV);

		// if(mode == 8)
		// {
		// 	cv::imshow("w", finalImageMean);
		// }

		Mat finalImageMedian;
		std::nth_element(pointsInsideContour.begin(), pointsInsideContour.begin() + pointsInsideContour.size() / 2, pointsInsideContour.end());

		thresholdValue = pointsInsideContour[pointsInsideContour.size() / 2] + 10;
		cv::threshold(grayscale, finalImageMedian, thresholdValue, 255, cv::THRESH_BINARY_INV);

		if(mode == 9)
		{
			cv::imshow("w", finalImageMedian);
		}

		frameCount++;

		auto now = high_resolution_clock::now();
		int elapsed = duration_cast<milliseconds>(now - beginTime).count();

		if (elapsed > 1000)
		{
			std::cout << frameCount << std::endl;
			frameCount = 0;
			beginTime = high_resolution_clock::now();
		}

		pressedKey = cv::waitKey(1);

		if (pressedKey >= '1' && pressedKey <= '9')
		{
			mode = pressedKey - '0';
		}
		else if (pressedKey == ' ')
		{
			paused = !paused;
		}
        else if (pressedKey == 'q')
		{
            return 1;
		}
		else if (pressedKey == 'b')
		{
			boarderLines = !boarderLines;
		}
		else if(pressedKey >= 'a' && pressedKey <= 'z')
		{
			if(pressedKey == 'a' || pressedKey == 'b')
				innerMode = pressedKey;
			else
				mode = pressedKey;
		}
    }
}