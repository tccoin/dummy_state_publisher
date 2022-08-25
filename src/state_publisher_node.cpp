#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <inekf_msgs/State.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <string>

#include "depth_generator.hpp"

using namespace std;

ros::Publisher pubState, pubTrajectoryGT, pubLeftImage, pubRightImage, pubDepth;
inekf_msgs::State state;
shared_ptr<ros::NodeHandle> nh_ptr;

bool trajectoryInitialized = false;
double stateCov;

/* TUM-BAG*/
void handle_pose(geometry_msgs::PoseStamped::Ptr msg) {
  // pose to inekf state
  state.header = msg->header;
  state.position = msg->pose.position;
  state.orientation = msg->pose.orientation;
  for (int i = 0; i < 9; ++i) {
    state.covariance[9 * i + i] = stateCov;
  }
  pubState.publish(state);

  // align backend trajectory to gt
  static tf2_ros::Buffer tfBuffer;
  static tf2_ros::TransformListener tfListener(tfBuffer);
  static tf2_ros::TransformBroadcaster br;
  static geometry_msgs::TransformStamped transformStampedPub;
  if (!trajectoryInitialized) {
    try {
      transformStampedPub =
          tfBuffer.lookupTransform("world", "kinect", ros::Time(0));
      transformStampedPub.child_frame_id = "trajectory";
    } catch (tf2::TransformException &ex) {
      return;
    }
  }
  transformStampedPub.header = msg->header;
  br.sendTransform(transformStampedPub);
}

void handle_trajectory(nav_msgs::Path::Ptr msg) {
  trajectoryInitialized = true;

  // publish gt trajectory
  static tf2_ros::Buffer tfBuffer;
  static tf2_ros::TransformListener tfListener(tfBuffer);
  static nav_msgs::Path trajectoryMsg;
  geometry_msgs::PoseStamped poseStamped;
  static geometry_msgs::TransformStamped transformStamped;
  try {
    transformStamped =
        tfBuffer.lookupTransform("world", "kinect", ros::Time(0));
  } catch (tf2::TransformException &ex) {
    return;
  }
  poseStamped.pose.orientation = transformStamped.transform.rotation;
  poseStamped.pose.position.x = transformStamped.transform.translation.x;
  poseStamped.pose.position.y = transformStamped.transform.translation.y;
  poseStamped.pose.position.z = transformStamped.transform.translation.z;
  trajectoryMsg.poses.push_back(poseStamped);
  trajectoryMsg.header = msg->header;
  trajectoryMsg.header.frame_id = "world";
  pubTrajectoryGT.publish(trajectoryMsg);
}

/* KITTI-RAW*/
queue<float> durations;
queue<vector<int>> odomPath;
string imageFolder, odomFile, gtFile, timeFile;
int imageIndex = 0;
shared_ptr<DepthGenerator> depthGenerator;
ros::Timer timerImage, timerImu;

void read_time() {
  ifstream file(timeFile);
  if (!file.is_open()) {
    ROS_ERROR_STREAM("Error when reading the time file.");
    return;
  }
  float lastTimestamp = 0;
  for (string line; getline(file, line);) {
    float timestamp = stof(line);
    durations.push(timestamp - lastTimestamp);
    lastTimestamp = timestamp;
  }
}

void read_odom() {
  ifstream file(odomFile);
  if (!file.is_open()) {
    ROS_ERROR_STREAM("Error when reading the odom file.");
    return;
  }
  for (string line; getline(file, line);) {
    istringstream in(line);
    odomPath.emplace((istream_iterator<float>(in)), istream_iterator<float>());
  }
}

void publish_images_and_odom(const ros::TimerEvent &e) {
  if (durations.empty()) {
    ROS_INFO_STREAM("end of dataset");
    timerImage.stop();
    return;
  } else {
    ROS_INFO_STREAM("output image");
  }

  // read stereo images with cv
  string fileName = to_string(imageIndex++);
  fileName = string(6 - fileName.length(), '0') + fileName + ".png";
  string leftFile = imageFolder + "/image_2/" + fileName;
  string rightFile = imageFolder + "/image_3/" + fileName;
  string depthFile = imageFolder + "/image_d/" + fileName;
  cv::Mat left = cv::imread(leftFile, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat right = cv::imread(rightFile, cv::ImreadModes::IMREAD_COLOR);
  if (left.data == nullptr || right.data == nullptr) {
    ROS_ERROR_STREAM("Loading image failed: " + fileName);
    return;
  }

  // calc depth
  ifstream f(depthFile.c_str());
  cv::Mat depth;
  if (f.good()) {
    depth = cv::imread(depthFile);
  } else {
    std::vector<float> leftDisparity;
    depthGenerator->disparity(left, right, leftDisparity);
    int height = left.rows;
    int width = left.cols;
    depth = cv::Mat(height, width, CV_16U);
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        Eigen::Vector3f result;
        auto status = depthGenerator->pt_depth_from_disparity(
            left, leftDisparity, j, i, result);
        if (status == TraceStatus::GOOD) {
          depth.at<ushort>(i, j) = result[2] * 100;
          // ROS_INFO_STREAM("depth: " << depth.at<ushort>(i, j) << ", "
          // << result[2] * 5000);
        } else {
          depth.at<ushort>(i, j) = std::numeric_limits<ushort>::infinity();
        }
      }
    }
    cv::imwrite(depthFile, depth);
  }

  // pose to inekf state
  if (odomPath.empty()) {
    ROS_ERROR_STREAM("No odom for this frame");
  }
  auto &path = odomPath.front();
  geometry_msgs::Point tmp;
  tf2::Vector3 pos(path[3], path[7], path[11]);
  state.position = tf2::toMsg(pos, tmp);
  tf2::Matrix3x3 rotationMatrix(path[0], path[1], path[2], path[4], path[5],
                                path[6], path[8], path[9], path[10]);
  tf2::Quaternion quad;
  rotationMatrix.getRotation(quad);
  state.orientation = tf2::toMsg(quad);
  for (int i = 0; i < 9; ++i) {
    state.covariance[9 * i + i] = stateCov;
  }
  odomPath.pop();

  // cv::Mat depth8U;
  // depth.convertTo(depth8U, CV_8U);
  // cv::imshow("depth", depth8U);
  // cv::waitKey(100);

  // publish
  std_msgs::Header header;
  header.seq = imageIndex;
  header.stamp = ros::Time::now();
  header.frame_id = "odom";
  pubLeftImage.publish(cv_bridge::CvImage(header, "bgr8", left).toImageMsg());
  pubRightImage.publish(cv_bridge::CvImage(header, "bgr8", right).toImageMsg());
  pubDepth.publish(cv_bridge::CvImage(header, "16UC1", depth).toImageMsg());
  state.header = header;
  pubState.publish(state);

  // if (imageIndex == 5) ros::shutdown();

  // sleep until next frame
  // if (durations.empty()) return;
  // ros::Duration duration(durations.front());
  durations.pop();
  // ros::Timer timerImage = nh_ptr->createTimer(duration, publish_images,
  // true);
}

void publish_imu(const ros::TimerEvent &e) { ROS_INFO_STREAM("output imu"); }

int main(int argc, char **argv) {
  ros::init(argc, argv, "state_publisher");
  ros::NodeHandle nh("~");
  // nh_ptr.reset(&nh);

  string datasetType;
  nh.param<string>("dataset_type", datasetType, "tum_ros");
  nh.param<double>("state_covariance", stateCov, 0.01);
  if (datasetType == "tum_ros") {
    string pose_topic, trajectory_topic;
    nh.param<string>("pose_topic", pose_topic, "pose_in");
    nh.param<string>("trajectory_topic", trajectory_topic, "trajectory_in");

    ros::Subscriber subPose = nh.subscribe(pose_topic, 2000, &handle_pose);
    ROS_INFO("subscribing: %s", subPose.getTopic().c_str());

    ros::Subscriber subTrajectory =
        nh.subscribe(trajectory_topic, 2000, &handle_trajectory);
    ROS_INFO("subscribing: %s", subTrajectory.getTopic().c_str());
  } else if (datasetType == "kitti_raw") {
    // load params
    float baseline, fu, fv, cx, cy;
    Eigen::Matrix3f intrinsic;
    nh.param<string>("image_folder", imageFolder, "");
    nh.param<string>("odom_file", odomFile, "");
    nh.param<string>("gt_file", gtFile, "");
    nh.param<string>("time_file", timeFile, "");
    nh.param<float>("camera_baseline", baseline, 0.54);
    nh.param<float>("camera_fu", fu, 707.0912);
    nh.param<float>("camera_fv", fv, 707.0912);
    nh.param<float>("camera_cx", cx, 601.8873);
    nh.param<float>("camera_cy", cy, 183.1104);
    intrinsic << fu, 0, cx, 0, fv, cy, 0, 0, 1;
    depthGenerator.reset(new DepthGenerator(baseline, intrinsic));
    // setup publishers
    pubLeftImage = nh.advertise<sensor_msgs::Image>("color_left", 10);
    pubRightImage = nh.advertise<sensor_msgs::Image>("color_right", 10);
    pubDepth = nh.advertise<sensor_msgs::Image>("depth", 10);
    // read files
    read_time();
    read_odom();
    // publish images
    ros::Duration duration(0.1);
    durations.pop();
    timerImage = nh.createTimer(duration, publish_images_and_odom);
    // publish imu
    // timerImu = nh.createTimer(ros::Duration(0.05), publish_imu);
  }

  pubState = nh.advertise<inekf_msgs::State>("state", 10);
  pubTrajectoryGT = nh.advertise<nav_msgs::Path>("gt_trajectory", 10);

  ros::spin();
}