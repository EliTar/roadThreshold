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

		Mat drawing = Mat::zeros(cannyDetection.size(), CV_8UC3);

		Mat cannyDetectionClone = cannyDetection.clone();

		RNG rng(12345);
        int rows = cannyDetectionClone.size[0] - 1;
		int cols = cannyDetectionClone.size[1] - 1;

		vector<vector<Point>> contours;
		vector<vector<Point>> smallContours;
		double contSum = 0;

		cv::findContours(cannyDetectionClone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		// This function changes its input!

		for(int i = 0; i < contours.size(); i++)
		{
			contSum += cv::contourArea(contours[i]);
		}

		double contAvg = contSum / contours.size();

		for(int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));

			double currArea = cv::contourArea(contours[i]);
			if(currArea < contAvg / 3)
			{
				cv::drawContours(drawing, contours, i, color);
				smallContours.push_back(contours[i]);
			}
		}

		if(mode == 4)
		{
			cv::imshow("w", drawing);
		}

		// Mat newCannyDetection = cannyDetection.clone();

		for(int i = 0; i < smallContours.size(); i++)
		{
			cv::drawContours(cannyDetection, smallContours, i, Scalar(0, 0, 0), -1);
		}

		cv::dilate(cannyDetection, cannyDetection, Mat());

		if(mode == 5)
		{
			cv::imshow("w", cannyDetection);
		}

		for(int i = 0;  i < rows; i++)
		{
			for(int t = 0; t < 3; t++)
			{
				cannyDetection.at<uchar>(i, t) = (uchar)0;
				cannyDetection.at<uchar>(i, cols - t) = (uchar)0;
			}
		}

		for(int i = 0;  i < cols; i++)
		{
			for(int t = 0; t < 3; t++)
			{
				cannyDetection.at<uchar>(t, i) = (uchar)0;
				cannyDetection.at<uchar>(rows - t, i) = (uchar)0;
			}
		}

		Mat cannyDetectionBlured;
		cv::blur(cannyDetection, cannyDetectionBlured, Size{8, 8});

		if(mode == 6)
		{
			cv::imshow("w", cannyDetectionBlured);
		}

		Mat biggest = Mat::zeros(cannyDetectionBlured.size(), CV_8U);

		int chosenContour = 0;

		int maxArea = 0;

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

		// Make sure we start fresh now with the canny detection...
		contours.clear();
		drawing = Mat::zeros(cannyDetection.size(), CV_8UC3);
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

		if(mode == 7)
		{
			cv::imshow("w", drawing);
		}

		if(mode == 8)
		{
			cv::imshow("w", biggest);
		}

		Mat maskedBiggest = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
		bitwise_and(grayscale, biggest, maskedBiggest);

		if(mode == 9)
		{
				cv::imshow("w", maskedBiggest);
		}

		vector<uchar> valuesToThreshold;

		for(int i = 0; i < maskedBiggest.rows; i++)
		{
			for(int j = 0; j < maskedBiggest.cols; j++)
			{
				if(maskedBiggest.at<uchar>(i, j) > 0)
					valuesToThreshold.push_back(maskedBiggest.at<uchar>(i, j));
			}
		}

		double optimalThresholdByContour = cv::threshold(valuesToThreshold, valuesToThreshold, 100, 200, cv::THRESH_OTSU);

		// if(mode == 'v')
		// {
		// 	Rect drawRect = cv::boundingRect(contours[chosenContour]);
		// 	rectangle(biggest, drawRect, Scalar(255));
		// 	cv::imshow("w", biggest);
		// }

		// TODO: look at this function. it crashes when everything is blurry...

		try
		{
			cv::RotatedRect drawRect = cv::minAreaRect(contours[chosenContour]);
			cv::Point2f vertices[4];
			drawRect.points(vertices);

			vector<Point> dummyVertices;
			dummyVertices.push_back(vertices[0]);
			dummyVertices.push_back(vertices[1]);
			dummyVertices.push_back(vertices[2]);
			dummyVertices.push_back(vertices[3]);

			if(mode == 'q')
			{
				for (int i = 0; i < 4; i++)
					line(biggest, vertices[i], vertices[(i+1)%4], Scalar(255));

				cv::imshow("w", biggest);
			}

			vector<vector<Point> > contourDummy = vector<vector<Point>>(1, dummyVertices);
			cv::drawContours(biggest, contourDummy, 0, Scalar(255), -1);

			if(mode == 'w')
			{
				cv::imshow("w", biggest);
			}

			Mat rectMask = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
			bitwise_and(grayscale, biggest, rectMask);

			if(mode == 'e')
			{
				cv::imshow("w", rectMask);
			}

			vector<uchar> valuesToThresholdRect;

			for(int i = 0; i < maskedBiggest.rows; i++)
			{
				for(int j = 0; j < maskedBiggest.cols; j++)
				{
					if(biggest.at<uchar>(i, j) == 0)
						rectMask.at<uchar>(i, j) = 255;
					else
						valuesToThresholdRect.push_back(rectMask.at<uchar>(i, j));
				}
			}

			Mat thresholdByContour = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
			Mat thresholdByMask = Mat::zeros(cannyDetectionBlured.size(), CV_8U);

			cv::threshold(rectMask, thresholdByContour, optimalThresholdByContour, 200, cv::THRESH_BINARY);

			cout << valuesToThresholdRect.size() << " " << valuesToThreshold.size() << endl;

			double optimalRectThreshold = cv::threshold(valuesToThresholdRect, valuesToThresholdRect, 100, 200, cv::THRESH_OTSU);
			cv::threshold(rectMask, thresholdByMask, optimalRectThreshold, 200, cv::THRESH_BINARY);
			cout << optimalRectThreshold << " " << optimalThresholdByContour << endl;

			if(mode == 'r')
			{
				cv::imshow("w", thresholdByContour);
			}

			if(mode == 't')
			{
				cv::imshow("w", thresholdByMask);
			}

		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}


		// Mat newImage = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
		// bitwise_and(grayscale, biggest, newImage);

		// // if(mode == 7)
		// // {
		// // 	cv::imshow("w", newImage);
		// // }

		// //Mat_<CV_8U> newIm {newImage};

		// //cout << (int)grayscale.at<uchar>(0, 0) << endl;
		// //cout << newImage.size()<< endl;

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
        else if (pressedKey == 'z')
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