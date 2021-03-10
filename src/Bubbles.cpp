// Bubbles.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

void CountBubbles()
{
	cv::Mat src = cv::imread(R"(image_bubbles.jpg)", 1);
	cv::Mat src2;

	cv::resize(src, src, cv::Size(src.cols * 1.25, src.rows * 1.25), 0, 0, cv::INTER_LINEAR_EXACT);

	src.copyTo(src2);
	imshow("src", src);

	cvtColor(src2, src2, cv::COLOR_BGR2GRAY);

	cv::Mat blur;
	cv::Mat blur2;
	GaussianBlur(src, blur, cv::Size(3, 3), 1);
	GaussianBlur(src2, blur2, cv::Size(3, 3), 1);
	imshow("gauss", blur);

	std::vector<cv::Vec3f> circles;
	HoughCircles(blur2, circles, cv::HOUGH_GRADIENT, 1, 10, 100, 30, 1, 25);

	for (auto& elem : circles)
	{
		cv::Vec3i c = elem;
		const auto center = cv::Point(c[0], c[1]);
		circle(src2, center, 15, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
	}

	threshold(src2, src2, 20, 255, cv::THRESH_BINARY);
	bitwise_not(src2, src2);
	imshow("circles", src2);

	cv::Mat imgLaplacian;
	cv::Laplacian(blur, imgLaplacian, CV_8UC1);

	cv::Mat imgResult = blur - imgLaplacian;

	imshow("Laplace Filtered Image", imgLaplacian);
	imshow("New Sharpened Image", imgResult);

	GaussianBlur(imgResult, imgResult, cv::Size(3, 3), 1);

	cv::Mat binary;

	cvtColor(imgResult, binary, cv::COLOR_BGR2GRAY);
	adaptiveThreshold(binary, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 35, 0);
	imshow("Binary", binary);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
	cv::erode(binary, binary, element);

	cv::Mat cont;
	binary.copyTo(cont);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (auto i = 0; i < contours.size(); ++i)
	{
		if (contourArea(contours[i]) < 45)
		{
			drawContours(cont, contours, i, cv::Scalar(0), -1);
		}
	}

	bitwise_not(cont, cont);
	imshow("not", cont);

	cv::Mat dist;
	distanceTransform(cont, dist, cv::DIST_L2, 3);

	normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
	imshow("Distance Transform Image", dist);

	dist.convertTo(dist, CV_8UC1);
	adaptiveThreshold(dist, dist, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 35, 0);
	imshow("contours", dist);

	cv::Mat element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11), cv::Point(1, 1));
	cv::dilate(dist, dist, element2);
	imshow("dilate dist", dist);

	cv::Mat final = dist + src2;
	imshow("Combined mask", final);

	cv::Mat rgb;
	cvtColor(final, rgb, cv::COLOR_GRAY2RGB);

	cv::Mat mask;
	inRange(rgb, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);
	rgb.setTo(cv::Scalar(0, 165, 255), mask);

	cv::Mat bubbles = src + rgb;

	cv::dilate(final, final, element);
	imshow("dilate final", final);

	std::vector<std::vector<cv::Point>> contours3;
	cv::findContours(final, contours3, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	int index = 0;

	for (auto i = 0; i < contours3.size(); ++i)
	{
		drawContours(src, contours3, i, cv::Scalar(0, 165, 255), -1);
		double centerX = 0.0;
		double centerY = 0.0;

		for (const auto& p : contours3[i])
		{
			centerX += p.x;
			centerY += p.y;
		}

		index++;
		cv::putText(bubbles, std::to_string(index), cv::Point(centerX / contours3[i].size(), centerY / contours3[i].size()), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 165, 255), 2);
	}

	std::cout << "Bubble count: " << index << std::endl;
	imshow("Found bubbles", bubbles);

	cv::waitKey();
}

void FindSimilarBubbles()
{
	cv::Mat bubble = cv::imread(R"(bubble.png)", 1);
	cv::Mat src = cv::imread(R"(image_bubbles.jpg)", 1);
	cv::Mat original;
	src.copyTo(original);

	cvtColor(bubble, bubble, cv::COLOR_BGR2GRAY);
	cvtColor(src, src, cv::COLOR_BGR2GRAY);

	GaussianBlur(bubble, bubble, cv::Size(3, 3), 1);
	GaussianBlur(src, src, cv::Size(3, 3), 1);

	adaptiveThreshold(bubble, bubble, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 35, 0);
	adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 35, 0);

	cv::Mat filtered;

	bubble.convertTo(bubble, CV_32F);
	float bubbleArray[38][36];

	for (int i = 0; i < bubble.rows; ++i)
	{
		for (int j = 0; j < bubble.cols; ++j)
		{
			if (bubble.at<float>(i, j) == 0)
				bubbleArray[i][j] = -2;
			else if (bubble.at<float>(i, j) == 255)
				bubbleArray[i][j] = 1;
			else
				bubbleArray[i][j] = bubble.at<float>(i, j);
		}
	}

	cv::Mat kernel = cv::Mat(38, 36, CV_32F, bubbleArray);

	cv::filter2D(src, filtered, CV_8UC1, kernel);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(filtered, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	cvtColor(filtered, filtered, cv::COLOR_GRAY2RGB);
	int index = 0;

	for (auto i = 0; i < contours.size(); ++i)
	{
		drawContours(original, contours, i, cv::Scalar(0, 165, 255), -1);

		double centerX = 0.0;
		double centerY = 0.0;

		for (const auto& p : contours[i])
		{
			centerX += p.x;
			centerY += p.y;
		}

		index++;
		cv::putText(original, std::to_string(index), cv::Point(centerX / contours[i].size(), centerY / contours[i].size()), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 165, 255), 0.5);
	}

	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
	cv::dilate(filtered, filtered, element);

	imshow("bubble", bubble);
	imshow("src", src);
	imshow("Similar", filtered);
	imshow("final", original);

	cv::waitKey();
}

int main()
{
	CountBubbles();
	FindSimilarBubbles();

    return 0;
}
