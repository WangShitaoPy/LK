#include <iostream>
#include <ctype.h>
#include<vector>
#include <math.h>
#include<time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int calcuXY(Mat & _img, Point & _pt, Mat & _mat);
void calcuT(Mat & _img1, Mat & _img2, Point &_pt, Mat & _mat,int size);
void calcuUV(Mat &_mat_xy, Mat & _mat_t, Mat &_mat_uv);

int main()
{
	//VideoCapture cap(0);
	int count = 0;
	Mat frame, frame1, frame2,img1,img2;
	vector<vector<Point>> pt;
	//frame = imread("1.jpg");
	//cap.read(frame1);
	int grid_step = 10;
	int grid_size = 3;
	frame1 = imread("frame08.png");
	frame2 = imread("frame09.png");
	Mat mat_A=Mat::zeros(grid_size*grid_size,2,CV_32FC1);
	Mat mat_B=Mat::zeros(grid_size*grid_size, 1, CV_32FC1);
	Mat mat_UV = Mat::zeros(2, 1, CV_32FC1);

	for (int i = 0; i < (frame1.rows - grid_step) / grid_step; i++)
	{
		vector<Point> _pt;
		for (int j = 0; j < (frame1.cols - grid_step) / grid_step; j++)
		{
			_pt.push_back(Point((j + 1)*grid_step, (i + 1)*grid_step));
			//cout << _pt[j] << " ";
		}
		pt.push_back(_pt);
		//cout << endl;
	}
	
	/*while (cap.isOpened())
	{*/
		Point optflow(0,0);
		//cap.read(frame2);
		cvtColor(frame1, img1, CV_RGB2GRAY);
		cvtColor(frame2, img2, CV_RGB2GRAY);
		/*1.求传入点的空间梯度矩阵 3*3  返回9*2的矩阵
		calcuXY(Mat & _img, Point & _pt, Mat & _mat)

			2求传入点的灰度时间梯度矩阵 3 * 3 返回9 * 1的矩阵
			calcuT(Mat & _img1, Mat & _img2, Point &_pt, Mat & _mat)

			3求解传入点的UV，返回点的UV值, 2 * 1
			calcuUV(Mat &_mat_xy, Mat & _mat_t, Mat &_mat_uv)

			注：传入的Mat 为单通道的图像，传入点为3 * 3图像块顶点*/

		for (int x = 0; x < pt.size(); x++)
		{
			for (int y = 0; y < pt[0].size(); y++)
			{
				//cout << pt[x][y] << endl;
				calcuXY(img1, pt[x][y], mat_A);
				calcuT(img1, img2, pt[x][y], mat_B,grid_size);
				calcuUV(mat_A,mat_B,mat_UV);
				//optflow.x += mat_UV.at<float>(0, 0);
				//optflow.y += mat_UV.at<float>(1, 0);
				
				if ((mat_UV.at<float>(0, 0) < 20 && mat_UV.at<float>(0, 0) > 1) || (mat_UV.at<float>(1, 0) < 20 && mat_UV.at<float>(1, 0) > 1))
				{

					line(frame2, pt[x][y], Point(pt[x][y].x + mat_UV.at<float>(0, 0), pt[x][y].y + mat_UV.at<float>(1, 0)), Scalar(255, 0, 0), 2, 8);

				}
			}
			//cout << endl;
		}

		
		imshow("optical flow",frame2);
		imshow("optical f",frame1);
		waitKey(0);
		frame1 = frame2.clone();
	
	return 0;
}
void calcuT(Mat &_img1, Mat &_img2, Point & _pt, Mat &_mat,int size)
{
	Rect roi(Point(_pt.x, _pt.y), Point(_pt.x + size, _pt.y + size));
	Mat img1 = _img1(roi);
	Mat img2 = _img2(roi);
	Mat Sub = Mat::zeros(size, size, CV_32FC1);
	Mat Re = Mat::zeros(size*size, 1, CV_32FC1);
	int rowNum = img1.rows;
	int colNum = img1.cols;
	for (int i = 0; i < rowNum; i++)
	{
		uchar* data1 = img1.ptr<uchar>(i);
		uchar* data2 = img2.ptr<uchar>(i);
		float* output = Sub.ptr<float>(i);

		for (int j = 0; j < colNum; j++)
		{
			output[j] = (float)(data1[j] - data2[j]);
		}
	}
	int x = 0;
	float* R = Re.ptr<float>(x);
	for (int i = 0; i < rowNum; i++)
	{
		float* output = Sub.ptr<float>(i);
		for (int j = 0; j < colNum; j++)
		{
			R[x++] = output[j];
		}

	}
	_mat = Re.clone();
}
void calcuUV(Mat & _mat_xy, Mat & _mat_t, Mat & _mat_uv)
{
	Mat At = _mat_xy.t();
	Mat Ap = At*_mat_xy;
	invert(Ap, Ap);
	_mat_uv = Ap*At*(_mat_t);
}
int calcuXY(Mat &_img, Point &_pt, Mat &_mat)
{
	int x = _pt.x;
	int y = _pt.y;
	Rect selection(_pt.x - 1, _pt.y - 1, 5, 5);
	Mat selectionMat = _img(selection);
	int xLoop = 3;
	int yLoop = 3;

	for (int i = 0; i < xLoop; i++)
	{
		uchar *dataSelection = selectionMat.ptr<uchar>(i);

		for (int j = 0; j < yLoop; j++)
		{
			float *dataMat = _mat.ptr<float>(i * 3 + j);
			if ((i) < 0 || (i + 2) > selectionMat.rows)
				return -1;
			else
				dataMat[1] = (selectionMat.at<Vec<char, 1>>(i + 1, j + 1)[0] - selectionMat.at<Vec<char, 1>>(i, j + 1)[0]) / 1;

			if ((j) < 0 || (j + 2) > selectionMat.cols)
				return -1;
			else
				dataMat[0] = (selectionMat.at<Vec<char, 1>>(i + 1, j + 1)[0] - selectionMat.at<Vec<char, 1>>(i + 1, j)[0]) / 1;
		}
	}
	return 0;
}
//int calcuXY(Mat &_img, Point &_pt, Mat &_mat, int size)
//{
//	int x = _pt.x;
//	int y = _pt.y;
//	Rect selection(_pt.x - 1, _pt.y - 1, size + 2, size + 2);
//	Mat selectionMat = _img(selection);
//	//	Mat  hsv = Mat::zeros(frame.size(), CV_8UC3), hist = Mat::zeros(1, 20, CV_8UC1), backproj;
//	/*Rect selection;
//	selection.x = _pt.x;
//	selection.y = _pt.y;
//	selection.width = 3;
//	selection.height = 3;*/
//	int xLoop = size;
//	int yLoop = size;
//
//	for (int i = 0; i < xLoop; i++)
//	{
//		uchar *dataSelection = selectionMat.ptr<uchar>(i);
//
//		for (int j = 0; j < yLoop; j++)
//		{
//			float *dataMat = _mat.ptr<float>(i * size + j);
//			if ((i)<0 || (i + 2)>selectionMat.rows)
//				return -1;
//			else
//				dataMat[1] = (selectionMat.at<Vec<char, 1>>(i + 2, j + 1)[0] - selectionMat.at<Vec<char, 1>>(i, j + 1)[0]) / 2.0;
//
//			if ((j)<0 || (j + 2)>selectionMat.cols)
//				return -1;
//			else
//				dataMat[0] = (selectionMat.at<Vec<char, 1>>(i + 1, j + 2)[0] - selectionMat.at<Vec<char, 1>>(i + 1, j)[0]) / 2.0;
//		}
//	}
//	return 0;
//}
