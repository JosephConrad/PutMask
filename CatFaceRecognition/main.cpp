//
//  main.cpp
//  CatFaceRecognition
//
//  Created by Konrad Lisiecki on 16/01/15.
//  Copyright (c) 2015 Konrad Lisiecki. All rights reserved.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include<pthread.h>
#include<stdlib.h>
#include<unistd.h>

double alpha; /**< Simple contrast control */
int beta;  /**< Simple brightness control */

using namespace std;
using namespace cv;

double min_face_size=200;
double max_face_size=2000;
Mat mask;
int SIZE = 1024;

#pragma pack(push, 2)
struct RGB {        //members are in "bgr" order!
    uchar blue;
    uchar green;
    uchar red;
};

pthread_t tid[2];
int counter;
pthread_mutex_t mutexx = PTHREAD_MUTEX_INITIALIZER;
std::string imageName = "./images/pilot.jpg";
const char* masks[] = {"pig","pilot", "vendetta"};
std::vector<std::string> availableMasks(masks, masks + 3);
bool stop = false;
const char* modes[] = {"normal", "sepida", "gray"};
std::vector<std::string> availableModes(modes, modes + 3);
std::string mode = "normal";



Mat detectFace(Mat src);
Mat putMask(Mat src,Point center,Size face_size);



bool stopProgram(std::string input) {
    return input == "stop";
}

bool validName(std::string input)
{
    return std::find(availableMasks.begin(), availableMasks.end(), input) != availableMasks.end();
}

void* doSomeThing(void *arg)
{
    using namespace std;
    string input;
    
    while (1) {
        cout << "Please, enter mask name (stop to exit program): \n";
        cout << "Available masks: \n";
        for (int i = 0; i < availableMasks.size(); i ++) {
            cout << i+1 << ". "+ availableMasks.at(i) << endl;
        }
        getline (std::cin,input);
        
        if (validName(input)) {
            pthread_mutex_lock(&mutexx);
            cout << "You have changed to " << input << " mask\n";
            std::string ext;
            input == "horse" ? ext = "png" : ext = "jpg";
            imageName = "./images/" + input + "." + ext;
            pthread_mutex_unlock(&mutexx);
        } else if (stopProgram(input)) {
            stop = true;
        } else {
            cout << "Invalid name!\n";
        }
    }
    return NULL;
}
Mat grayScale(Mat frame){
    Mat gray = frame.clone();
    for (int i= 0; i<frame.rows; ++i)
    {
        for (int j = 0 ; j < frame.cols; ++j)
        {
            int b = frame.ptr<RGB>(i)[j].blue;
            int r = frame.ptr<RGB>(i)[j].red;
            int g = frame.ptr<RGB>(i)[j].green;
            int grayscale = ((r*77)+(b*28)+(g*151))/256;
            gray.ptr<RGB>(i)[j].blue = grayscale;
            gray.ptr<RGB>(i)[j].red = grayscale;
            gray.ptr<RGB>(i)[j].green = grayscale;
        }
    }
    // cv::Mat gray(frame.size(), CV_8UC1);
    // cvtColor(frame, gray, CV_BGR2Luv);
    return gray;
}

Mat sepia(Mat frame){
    Mat sepia = frame.clone();
    
    Mat kern = (cv::Mat_<float>(4,4) <<  0.272, 0.534, 0.131, 0,
                    0.349, 0.686, 0.168, 0,
                    0.393, 0.769, 0.189, 0,
                    0, 0, 0, 1);
    
    cv::transform(frame, sepia, kern);
    return sepia;
}


int main( )
{
    int err;
    err = pthread_create(&(tid[0]), NULL, &doSomeThing, NULL);
    if (err != 0)
        printf("\ncan't create thread :[%s]", strerror(err));
    
    VideoCapture cap(0);
    namedWindow( "window1", 1 );
    
    while(!stop)
    {
        mask = imread(imageName);
        Mat frame;
        cap >> frame;
        
        cv::Mat imgDistance(frame.size(), CV_8UC1);
        frame=detectFace(frame);

        if (mode == "sepia") {
            imshow( "window1", sepia(frame));
        } else if (mode == "gray") {
            imshow( "window1", grayScale(frame));
        } else
            imshow( "window1", frame);

        waitKey(1);
    }
    
    //waitKey(0);
    return 0;
}

Mat detectFace(Mat image)
{
    // Load Face cascade (.xml file)
    CascadeClassifier face_cascade( "/Users/konrad/Dropbox/09_semestr/obrazy/haarcascade_frontalface_default.xml" );
    
    // Detect faces
    std::vector<Rect> faces;
    
    face_cascade.detectMultiScale( image, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );
    
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {   // Lets only track the first face, i.e. face[0]
        min_face_size = faces[0].width*0.7;
        max_face_size = faces[0].width*1.5;
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        image=putMask(image,center,Size( faces[i].width, faces[i].height));
    }
    return image;
}

Mat putMask(Mat src,Point center,Size face_size)
{
    Mat mask1,src1;
    resize(mask,mask1,face_size);
    
    // ROI selection
    Rect roi(center.x - face_size.width/2, center.y - face_size.width/2, face_size.width, face_size.width);
    src(roi).copyTo(src1);
    
    // to make the white region transparent
    Mat mask2,m,m1;
    cvtColor(mask1,mask2,CV_BGR2GRAY);
    threshold(mask2,mask2,230,255,CV_THRESH_BINARY_INV);
    
    vector<Mat> maskChannels(3),result_mask(3);
    split(mask1, maskChannels);
    bitwise_and(maskChannels[0],mask2,result_mask[0]);
    bitwise_and(maskChannels[1],mask2,result_mask[1]);
    bitwise_and(maskChannels[2],mask2,result_mask[2]);
    merge(result_mask,m );         //    imshow("m",m);
    
    mask2 = 255 - mask2;
    vector<Mat> srcChannels(3);
    split(src1, srcChannels);
    bitwise_and(srcChannels[0],mask2,result_mask[0]);
    bitwise_and(srcChannels[1],mask2,result_mask[1]);
    bitwise_and(srcChannels[2],mask2,result_mask[2]);
    merge(result_mask,m1 );        //    imshow("m1",m1);
    
    addWeighted(m,1,m1,1,0,m1);    //    imshow("m2",m1);
    
    m1.copyTo(src(roi));
    
    return src;
}