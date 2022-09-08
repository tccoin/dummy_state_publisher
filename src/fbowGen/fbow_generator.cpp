#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <inekf_msgs/State.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <fbow/fbow.h>
#include <fbow/vocabulary_creator.h>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <string>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
using namespace std;
/* KITTI-RAW*/
queue<float> durations;
queue<vector<float>> odomPath, gtPath;
string imageFolder, odomFile, gtFile, timeFile;
int imageIndex = 0;
ros::Timer timerImage, timerImu;
nav_msgs::Path odomTrajectoryMsg, gtTrajectoryMsg;
int feature_type;
vector<cv::Mat> descriptors;

bool readAndDetect() {
 
  // read stereo images with cv
  string fileName = to_string(imageIndex++);
  fileName = string(6 - fileName.length(), '0') + fileName + ".png";
  string leftFile = imageFolder +"/" + fileName;
  string depthFile = imageFolder + "/image_d/" + fileName;
  if (imageIndex == 4500){
return false;
  }
  cv::Mat left = cv::imread(leftFile, cv::ImreadModes::IMREAD_COLOR);
  if (left.data == nullptr) {
    ROS_ERROR_STREAM("Loading image failed: " + fileName);
    return false;
  }
  cv::Ptr<cv::Feature2D> fdetector;
  if (feature_type== 0)   fdetector=cv::ORB::create(2000);
  if (feature_type== 1)  fdetector=cv::SIFT::create(2000);
  vector<cv::KeyPoint> keypoint;
  cv::Mat descriptor;
  fdetector->detectAndCompute(left, cv::Mat(), keypoint, descriptor);
  descriptors.push_back(descriptor);
  ROS_INFO_STREAM("proecessing " << imageIndex);



  // if (imageIndex == 5) ros::shutdown();

  // sleep until next frame
  // if (durations.empty()) return;
  // ros::Duration duration(durations.front());
    return true;
  // ros::Timer timerImage = nh_ptr->createTimer(duration, publish_images,
  // true);
}

void publish_imu(const ros::TimerEvent &e) { ROS_INFO_STREAM("output imu"); }

int main(int argc, char **argv) {
  ros::init(argc, argv, "state_publisher");
  ros::NodeHandle nh("~");
  // nh_ptr.reset(&nh)
   nh.param<string>("image_folder", imageFolder, "");
    string vocab_path;
    nh.param<string>("vocab_path", vocab_path, "");
    bool readExistingVocab; 
    nh.param<bool>("read_existing_vocab",readExistingVocab,false);
    nh.param<int>("feature_type",feature_type,1); //0 for orb 1 for sift
    // publish images 
    bool flag = true;
    do{
      flag = readAndDetect();
    }while(flag);
    // publish imu
    // timerImu = nh.createTimer(ros::Duration(0.05), publish_imu);
    fbow::Vocabulary fvoc;
    auto t_start=std::chrono::high_resolution_clock::now();
    if (readExistingVocab){
      fvoc.readFromFile(vocab_path);
    }else{
      fbow::VocabularyCreator::Params params;
      params.k = 10;
      params.L = 5;
      params.nthreads=1;
      params.maxIters=0;  
      fbow::VocabularyCreator vocabCat;
      vocabCat.create(fvoc,descriptors,"hf-net",params);

    }
    fvoc.saveToFile("/home/bigby/ws/catkin_ws/siftKitt.fbow");
    ROS_INFO_STREAM("complete  ======" );
    ros::spin();
}