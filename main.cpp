//dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include<vector>
#include<string>

cv::Rect face_detect_dlib(cv::Mat &color);
std::vector<cv::Point2f> face_landmark_detect(cv::Rect &rect, cv::Mat &color);

int main(int argc,char** argv)
{
	
	cv::Mat Image;
	Image = cv::imread(argv[1]);
	cv::Rect rect = face_detect_dlib(Image);
	std::vector<cv::Point2f> landmarks = face_landmark_detect(rect, Image);
	return 0;
}

//dlib检测人脸框
cv::Rect face_detect_dlib(cv::Mat &color)
{
//将图片缩小，提高速度
#define FACTOR 3
	cv::Mat tmp;
	cv::resize(color, tmp, color.size() / FACTOR);
	dlib::cv_image<dlib::rgb_pixel> dlib_img(tmp);
	dlib::frontal_face_detector detector =dlib::get_frontal_face_detector();;
	std::vector<dlib::rectangle> dets = detector(dlib_img);

	if (dets.size() == 0)
	{
		//cout << "dlib检测失败" << endl;
		return cv::Rect();
	}
	else
	{
		dlib::rectangle max_det(0, 0, 0, 0);
		for (unsigned short i = 0; i < dets.size(); i++)
		{
			if (dets[i].area() > max_det.area())
			{
				max_det = dets[i];
			}
		}
		cv::Rect rect_dlib = cv::Rect(max_det.left() * FACTOR, max_det.top() * FACTOR, max_det.width() * FACTOR, max_det.height() * FACTOR);
		return rect_dlib;
	}
}

//dlib检测关键点
std::vector<cv::Point2f> face_landmark_detect(cv::Rect &rect, cv::Mat &color)
{
	cv::Mat gray;
	cv::Mat tmp = color.clone();
	dlib::shape_predictor sp;
	
	std::string modleRoot = "../../shape_predictor_68_face_landmarks.dat"; //模型
	dlib::deserialize(modleRoot) >> sp;

	cvtColor(color, gray, cv::COLOR_RGB2GRAY);
	dlib::cv_image<unsigned char> dlib_img(gray);
	dlib::rectangle face(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);

	dlib::full_object_detection shapes = sp(dlib_img, face);
	std::vector<cv::Point2f> landmarks;
	for (int idx = 0;idx < shapes.num_parts();idx++)
	{
		cv::Point2f landmark(shapes.part(idx).x(), shapes.part(idx).y());
		landmarks.push_back(landmark);
	}
	for (int i = 0;i < landmarks.size();i++)
	{
		cv::rectangle(tmp, rect, cv::Scalar(0, 255, 0), 2);
		cv::circle(tmp, cv::Point2f(landmarks[i].x, landmarks[i].y), 3, cv::Scalar(0, 0, 255), -1, 8);
	}
	cv::imwrite("../../result.png", tmp);
	cv::imshow("hh",tmp);
	cv::waitKey(0);
	return landmarks;
}