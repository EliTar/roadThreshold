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

		// for(int i = 20;  i < grayscale.size[0] - 20; i++)
		// {
		// 	grayscale.at<uchar>(i, 0) = (uchar)255;
		// 	grayscale.at<uchar>(i, grayscale.size[1] - 1) = (uchar)255;
		// }

		// for(int i = 20;  i < grayscale.size[1] - 20; i++)
		// {
		// 	grayscale.at<uchar>(0, i) = (uchar)255;
		// 	grayscale.at<uchar>(grayscale.size[0] - 1, i) = (uchar)255;
		// }

		// grayscale.row(0).setTo(Scalar(255));
		// grayscale.row(grayscale.size[0] - 1).setTo(Scalar(255));
		// grayscale.col(0).setTo(Scalar(255));
		// grayscale.col(grayscale.size[1] - 1).setTo(Scalar(255));
		// cv::threshold(grayscale, currFrame, 80, 255, CV_THRESH_BINARY);

		if (mode == 1)
		{
			cv::imshow("w", grayscale);
		}

		Mat blurred;
		Mat diff;
		//cv::blur(grayscale, blurred, Size{13, 13});
		cv::blur(grayscale, blurred, Size{5, 5});
		// blurred = cv::Scalar::all(255) - blurred;

		if(innerMode == 'b')
		{
			cv::blur(grayscale, blurred, Size{10, 10});
			cv::subtract(grayscale, blurred, diff);
			cv::blur(diff, diff, Size{5, 5});
			cv::dilate(diff, diff, Mat(), Point(-1,-1), 3);
		}

		if (mode == 2)
		{
			if(innerMode == 'a')
				cv::imshow("w", blurred);
			if(innerMode == 'b')
				cv::imshow("w", diff);
		}



		// for(int i = range;  i < blurred.size[0] - range; i++)
		// {
		// 	blurred.at<uchar>(i, 0) = (uchar)255;
		// 	blurred.at<uchar>(i, blurred.size[1] - 1) = (uchar)255;
		// }

		// for(int i = range;  i < blurred.size[1] - range; i++)
		// {
		// 	blurred.at<uchar>(0, i) = (uchar)255;
		// 	blurred.at<uchar>(blurred.size[0] - 1, i) = (uchar)255;
		// }

		// if (mode == 3)
		// {
		// 	Mat edgeDetection;
		// 	cv::blur(foundColors, edgeDetection, Size{13, 13});
		// 	cv::Canny(edgeDetection, edgeDetection, 70, 110);
		// 	cv::imshow("w", edgeDetection);
		// }

		Mat cannyDetection;

		if(innerMode == 'a')
			cv::Canny(blurred, cannyDetection, cannyLowThreshold, cannyHighThreshold);
			cv::dilate(cannyDetection, cannyDetection, Mat());
		if(innerMode == 'b')
		{
 			cv::Canny(diff, cannyDetection, cannyLowThreshold, cannyHighThreshold);
			cv::dilate(cannyDetection, cannyDetection, Mat(), Point(-1,-1), 3);
		}

		if (mode == 3)
		{
			cv::imshow("w", cannyDetection);
			cv::createTrackbar("Low Threshold", "w", &cannyLowThreshold, 200);
			cv::createTrackbar("High Threshold", "w", &cannyHighThreshold, 200);
		}

		Mat cannyDetectionBlured;
		cv::blur(cannyDetection, cannyDetectionBlured, Size{8, 8});

		// for(int i = range;  i < cannyDetectionBlured.size[0] - range; i++)
		// {
		// 	cannyDetectionBlured.at<uchar>(i, 1) = (uchar)255;
		// 	cannyDetectionBlured.at<uchar>(i, cannyDetectionBlured.size[1] - 2) = (uchar)255;
		// }

		// for(int i = range;  i < cannyDetectionBlured.size[1] - range; i++)
		// {
		// 	cannyDetectionBlured.at<uchar>(1, i) = (uchar)255;
		// 	cannyDetectionBlured.at<uchar>(cannyDetectionBlured.size[0] - 2, i) = (uchar)255;
		// }

		// cv::dilate(cannyDetection, cannyDetection, Mat());


		if(mode == 4)
		{
			cv::imshow("w", cannyDetectionBlured);
		}

		if(mode == 'z')
		{
			Mat hough = cannyDetection.clone();
			vector<cv::Vec4i> lines;
			HoughLinesP(hough, lines, 1, CV_PI/thetaDivider, houghThreshold, minLineLength, maxLineGap);

			// cv::createTrackbar("RHO", "w", &rho, 3);
			cv::createTrackbar("thetaDivider", "w", &thetaDivider, 200);
			cv::createTrackbar("Threshold", "w", &houghThreshold, 300);
			cv::createTrackbar("minLineLength", "w", &minLineLength, 100);
			cv::createTrackbar("maxLineGap", "w", &maxLineGap, 100);

			for( size_t i = 0; i < lines.size(); i++ )
			{
				cv::Vec4i l = lines[i];
				line(hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
			}

			cv::imshow("w", hough);
		}

		if(mode == 'x')
		{
			Mat hough = cannyDetection.clone();
			vector<cv::Vec2f> lines;
			HoughLines(hough, lines, 1, CV_PI/180, houghThreshold, srn, stn);

			// cv::createTrackbar("RHO", "w", &rho, 3);
			// cv::createTrackbar("Theta", "w", &theta, 5);
			cv::createTrackbar("Threshold", "w", &houghThreshold, 300);
			cv::createTrackbar("srn", "w", &srn, 3);
			cv::createTrackbar("stn", "w", &stn, 3);

			for( size_t i = 0; i < lines.size(); i++ )
			{
				float rho = lines[i][0], theta = lines[i][1];
				Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
				line(hough, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
			}

			cv::imshow("w", hough);
		}

		Mat drawing = Mat::zeros(cannyDetectionBlured.size(), CV_8UC3);
		Mat biggest = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
		int chosenContour = 0;
		int pvContour = -1;
		int pvpvContour = -1;

		int maxArea = 0;
		RNG rng(12345);

		int jumpValue = 20;
		int boarderLengthRows = 400;
		int boarderLengthCols = 600;
		int corner = 10;

		// cout << cannyDetectionBlured.size[1] << cannyDetectionBlured.size[0] << endl;

		if(boarderLines)
		{
			// int rows = cannyDetectionBlured.size[0] - 1;
			// int cols = cannyDetectionBlured.size[1] - 1;
			// for(int i = corner + jumpValue;  i < rows - corner - jumpValue; i += boarderLengthRows + jumpValue)
			// {
			// 	for(int j = i; j < i + boarderLengthRows; j++)
			// 	{
			// 		cannyDetectionBlured.at<uchar>(j, 1) = (uchar)255;
			// 		cannyDetectionBlured.at<uchar>(j, cols - 1) = (uchar)255;
			// 	}
			// }
			// for(int i = corner + jumpValue;  i < cols - corner - jumpValue; i += boarderLengthCols + jumpValue)
			// {
			// 	for(int j = i; j < i + boarderLengthCols; j++)
			// 	{
			// 		cannyDetectionBlured.at<uchar>(1, j) = (uchar)255;
			// 		cannyDetectionBlured.at<uchar>(rows - 1, j) = (uchar)255;
			// 	}
			// }

			int rows = cannyDetectionBlured.size[0] - 1;
			int cols = cannyDetectionBlured.size[1] - 1;
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
				pvpvContour = pvContour;
				pvContour = chosenContour;
				maxArea = currArea;
				chosenContour = i;
			}
		}

		//Mat mask = Mat::zeros(cannyDetectionBlured.size(), CV_8U);

		// cv::drawContours(biggest, contours, chosenContour, Scalar(255), -1);

		vector<Point> newCont;

		int rows = cannyDetectionBlured.size[0] - 1;
		int cols = cannyDetectionBlured.size[1] - 1;

		for(Point p : contours[chosenContour])
		{
			if(p.x != 1 && p.y != 1 && p.y != rows - 1 && p.x != cols - 1)
			{
				newCont.push_back(p);
			}
			else
			{
				cout << p << endl;
			}
		}

		contours[chosenContour] = newCont;

		cv::drawContours(biggest, contours, chosenContour, Scalar(255), -1);

		//cv::drawContours(mask, contours, chosenContour, Scalar(1), -1);

		// vector<Point> biggestContour = contours[chosenContour];

		if(mode == 5)
		{
			// for (vector<Point2i> &c : contours) {
			// 	double currArea = cv::contourArea(c);
			// 	if (currArea > maxArea)
			// 	{
			// 		maxArea = currArea;
			// 		chosenContour = c;
			// 	}
			// }

			cv::imshow("w", drawing);
		}

		// Mat element = getStructuringElement(cv::MORPH_RECT, Size(10, 10));
		// morphologyEx(biggest, biggest, cv::MORPH_CLOSE, element);

		if(mode == 6)
		{
			cv::imshow("w", biggest);
		}

		if(mode == 'v')
		{
			Rect drawRect = cv::boundingRect(contours[chosenContour]);
			rectangle(biggest, drawRect, Scalar(255));
			cv::imshow("w", biggest);
		}

		if(mode == 'n')
		{
			cv::RotatedRect drawRect = cv::minAreaRect(newCont);
			cv::Point2f vertices[4];
			drawRect.points(vertices);
			for (int i = 0; i < 4; i++)
				line(biggest, vertices[i], vertices[(i+1)%4], Scalar(255));

			cv::imshow("w", biggest);
		}

		Mat newImage = Mat::zeros(cannyDetectionBlured.size(), CV_8U);
		bitwise_and(grayscale, biggest, newImage);

		if(mode == 7)
		{
			cv::imshow("w", newImage);
		}

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

		if(mode == 8)
		{
			cv::imshow("w", finalImageMean);
		}

		Mat finalImageMedian;
		std::nth_element(pointsInsideContour.begin(), pointsInsideContour.begin() + pointsInsideContour.size() / 2, pointsInsideContour.end());
		//cout << "median is " << pointsInsideContour[pointsInsideContour.size() / 2] << endl;

		thresholdValue = pointsInsideContour[pointsInsideContour.size() / 2] + 10;
		cv::threshold(grayscale, finalImageMedian, thresholdValue, 255, cv::THRESH_BINARY_INV);

		if(mode == 9)
		{
			cv::imshow("w", finalImageMedian);
		}


		// for(int i = 0; i < currFrame.rows; i++)
		// {
		// 	for(int j = 0; j < currFrame.rows; j++)
		// 	{
		// 		Point point{i, j};
		// 		if(cv::pointPolygonTest(biggestContour, point, false))
		// 		{
		// 			insideContour.push_back(point);
		// 		}
		// 	}
		// }

		//std::cout << insideContour.size() << std::endl;


		frameCount++;

		auto now = high_resolution_clock::now();
		int elapsed = duration_cast<milliseconds>(now - beginTime).count();

		if (elapsed > 1000)
		{
			std::cout << frameCount << std::endl;
			frameCount = 0;
			beginTime = high_resolution_clock::now();
		}

		// cv::imshow("Video feed", currFrame);

		pressedKey = cv::waitKey(1);

		if (pressedKey >= '1' && pressedKey <= '9')
		{
			mode = pressedKey - '0';
		}
		else if (pressedKey == ' ')
		{
			paused = !paused;
		}
		else if (pressedKey == 'm')
		{
			boarderLines = !boarderLines;
		}
		else if (pressedKey == 'q')
		{
			return 0;
		}
		else if(pressedKey >= 'a' && pressedKey <= 'z')
		{
			if(pressedKey == 'a' || pressedKey == 'b')
				innerMode = pressedKey;
			else
				mode = pressedKey;
		}
    }
    // return 0;
}

		// Mat largeDiff;
		// grayscale.convertTo(largeDiff, CV_16U);
		// largeDiff += 127;
		// Mat diff;
		// cv::subtract(largeDiff, blurred, diff, cv::noArray(), CV_8U);

		// if (mode == 3)
		// {
		// 	cv::imshow("w", diff);
		// }

		// if (mode == 4)
		// {
		// 	Mat binary;
		// 	cv::threshold(diff, binary, 127, 255, cv::THRESH_BINARY);
		// 	cv::imshow("w", binary);
		// }

		// Mat black;
		// Mat white;

		// int minDiff = 10;

		// cv::threshold(diff, white, 127 + minDiff, 255, cv::THRESH_BINARY);
		// cv::threshold(diff, black, 127 - minDiff, 255, cv::THRESH_BINARY_INV);

		// Mat foundColors{currFrame.rows, currFrame.cols, CV_8U, Scalar{127}};
		// cv::bitwise_or(foundColors, white, foundColors);
		// cv::subtract(foundColors, black, foundColors);

		// if (mode == 5)
		// {
		// 	cv::imshow("w", foundColors);
		// }

		// Mat gray{currFrame.rows, currFrame.cols, CV_8U, 255};
		// gray -= black;
		// gray -= white;

		// if (mode == 6)
		// {
		// 	cv::imshow("w", gray);
		// }