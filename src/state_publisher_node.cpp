#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <inekf_msgs/State.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
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

ros::Publisher pubState, pubTrajectoryGT, pubOdomTrajectory, pubLeftImage,
    pubRightImage, pubDepth, pubCameraInfo, pubImu;
inekf_msgs::State state;
shared_ptr<ros::NodeHandle> nh_ptr;
string worldFrame;
Eigen::Matrix3f intrinsicG;
bool trajectoryInitialized = false;
double stateCov;
queue<float> durations;
queue<vector<float>> odomPath, gtPath, oxtsImuData;
string imageFolder, oxtsImuFolder, odomFile, gtFile, timeFile;
int imageIndex = 0;
float warmupDuration = 0;
float frameDuration = 0.16;
shared_ptr<DepthGenerator> depthGenerator;
ros::Timer timerImage, timerImu;
nav_msgs::Path odomTrajectoryMsg, gtTrajectoryMsg;

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
          tfBuffer.lookupTransform(worldFrame, "kinect", ros::Time(0));
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
        tfBuffer.lookupTransform(worldFrame, "kinect", ros::Time(0));
  } catch (tf2::TransformException &ex) {
    return;
  }
  poseStamped.pose.orientation = transformStamped.transform.rotation;
  poseStamped.pose.position.x = transformStamped.transform.translation.x;
  poseStamped.pose.position.y = transformStamped.transform.translation.y;
  poseStamped.pose.position.z = transformStamped.transform.translation.z;
  trajectoryMsg.poses.push_back(poseStamped);
  trajectoryMsg.header = msg->header;
  trajectoryMsg.header.frame_id = worldFrame;
  pubTrajectoryGT.publish(trajectoryMsg);
}

/* KITTI-RAW*/

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

void read_traj(string filePath, queue<vector<float>> &q) {
  ifstream file(filePath);
  if (!file.is_open()) {
    ROS_ERROR_STREAM("Error when reading the trajectory file.");
    return;
  }
  for (string line; getline(file, line);) {
    istringstream in(line);
    vector<float> nums;
    copy(istream_iterator<float>(in), istream_iterator<float>(),
         back_inserter(nums));
    q.push(nums);
  }
}

void read_oxts_imu(string folder, queue<vector<float>> &q) {
  for (int i = 0; i < durations.size(); i++) {
    string filePath = folder + '/' + string(10 - to_string(i).length(), '0') +
                      to_string(i) + ".txt";
    ifstream file(filePath);
    if (!file.is_open()) {
      ROS_ERROR_STREAM("Error when reading the imu file: " << filePath);
      return;
    }
    for (string line; getline(file, line);) {
      istringstream in(line);
      vector<float> nums;
      copy(istream_iterator<float>(in), istream_iterator<float>(),
           back_inserter(nums));
      q.push(nums);
    }
  }
}

tf2::Vector3 get_pos(vector<float> &transform) {
  return tf2::Vector3(transform[3], transform[7], transform[11]);
}

tf2::Matrix3x3 get_rot(vector<float> &transform) {
  return tf2::Matrix3x3(transform[0], transform[1], transform[2], transform[4],
                        transform[5], transform[6], transform[8], transform[9],
                        transform[10]);
}

tf2::Quaternion get_quad(vector<float> &transform) {
  auto rotationMatrix = get_rot(transform);
  tf2::Quaternion quad;
  rotationMatrix.getRotation(quad);
  return quad;
}
tf2::Quaternion get_quadV2(vector<float> &transform) {
  tf2::Quaternion quad(transform[3], transform[4], transform[5], transform[6]);
  return quad;
}
tf2::Vector3 get_posV2(vector<float> &transform) {
  return tf2::Vector3(transform[0], transform[1], transform[2]);
}
void publish_images_and_odom(const ros::TimerEvent &e) {
  if (durations.empty()) {
    ROS_INFO_STREAM("end of dataset");
    timerImage.stop();
    return;
  } else {
    // ROS_INFO_STREAM("output image");
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
    // set<float> depths;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        Eigen::Vector3f result;
        auto status = depthGenerator->pt_depth_from_disparity(
            left, leftDisparity, j, i, result);
        if (status == TraceStatus::GOOD) {
          // if (result[2] < 10) depths.insert(result[2]);
          depth.at<ushort>(i, j) = min(65.535f, float(result[2])) * 1000;
          // ROS_INFO_STREAM("depth: " << depth.at<ushort>(i, j) << ", "
          // << result[2] * 5000);
        } else {
          depth.at<ushort>(i, j) = 0;
        }
      }
    }
    // string result = "depths:";
    // for (float depth : depths) result += " " + to_string(depth);
    // ROS_INFO_STREAM(result);
    cv::imwrite(depthFile, depth);
  }

  // odom trajectory
  if (odomPath.empty()) {
    ROS_ERROR_STREAM("No odom for this frame");
  }
  auto &transform = odomPath.front();

  geometry_msgs::Point tmp;
  auto pos = get_pos(transform);
  state.position = tf2::toMsg(pos, tmp);
  auto rotationMatrix = get_rot(transform);

  tf2::Quaternion rot;
  rot.setEuler(-1.5707, 0, 0);
  tf2::Quaternion quad = get_quad(transform);
  quad = rot * quad;
  state.orientation = tf2::toMsg(quad);
  for (int i = 0; i < 9; ++i) {
    state.covariance[9 * i + i] = stateCov;
  }
  odomPath.pop();

  geometry_msgs::PoseStamped poseStamped;
  poseStamped.pose.orientation = state.orientation;
  poseStamped.pose.position = state.position;
  odomTrajectoryMsg.poses.push_back(poseStamped);

  // gt trajectory
  if (gtPath.empty()) {
    ROS_ERROR_STREAM("No gt for this frame");
  }
  auto &transformGT = gtPath.front();
  geometry_msgs::PoseStamped poseStampedGT;
  quad = get_quad(transformGT);
  poseStampedGT.pose.orientation = tf2::toMsg(quad);
  pos = get_pos(transformGT);
  poseStampedGT.pose.position = tf2::toMsg(pos, tmp);
  gtTrajectoryMsg.poses.push_back(poseStampedGT);
  gtPath.pop();

  if (false) {
    // visualize depth image
    cv::Mat depth8U;
    depth.convertTo(depth8U, CV_8U, 1.0 / 256);
    cv::imshow("depth dummy", depth8U);
    cv::waitKey(100);
  }
  // camera info
  sensor_msgs::CameraInfo cameraInfoMsg;
  auto intrinsic = depthGenerator->intrinsic;
  cameraInfoMsg.K = boost::array<double, 9>{
      intrinsic(0, 0), intrinsic(0, 1), intrinsic(0, 2),
      intrinsic(1, 0), intrinsic(1, 1), intrinsic(1, 2),
      intrinsic(2, 0), intrinsic(2, 1), intrinsic(2, 2)};
  cameraInfoMsg.height = left.rows;
  cameraInfoMsg.width = left.cols;

  // oxts imu
  auto &imuData = oxtsImuData.front();
  sensor_msgs::Imu imuMsg;
  quad.setRPY(imuData[3], imuData[4], imuData[5]);
  imuMsg.orientation = tf2::toMsg(quad);
  imuMsg.angular_velocity.x = imuData[17];
  imuMsg.angular_velocity.y = imuData[18];
  imuMsg.angular_velocity.z = imuData[19];
  imuMsg.linear_acceleration.x = imuData[11];
  imuMsg.linear_acceleration.y = imuData[12];
  imuMsg.linear_acceleration.z = imuData[13];
  oxtsImuData.pop();

  // publish
  std_msgs::Header header;
  header.seq = imageIndex;
  header.stamp = ros::Time::now();
  header.frame_id = worldFrame;
  pubLeftImage.publish(cv_bridge::CvImage(header, "bgr8", left).toImageMsg());
  pubRightImage.publish(cv_bridge::CvImage(header, "bgr8", right).toImageMsg());
  pubDepth.publish(cv_bridge::CvImage(header, "16UC1", depth).toImageMsg());
  state.header = header;
  pubState.publish(state);
  pubCameraInfo.publish(cameraInfoMsg);
  odomTrajectoryMsg.header = header;
  pubOdomTrajectory.publish(odomTrajectoryMsg);
  gtTrajectoryMsg.header = header;
  pubTrajectoryGT.publish(gtTrajectoryMsg);
  imuMsg.header = header;
  pubImu.publish(imuMsg);

  // if (imageIndex == 5) ros::shutdown();

  // sleep until next frame
  // if (durations.empty()) return;
  // ros::Duration duration(durations.front());
  durations.pop();
  // ros::Timer timerImage = nh_ptr->createTimer(duration, publish_images,
  // true);
}

void publish_images_and_odom_sc(const ros::TimerEvent &e) {
  if (odomPath.empty()) {
    ROS_INFO_STREAM("End of Dataset.");
    exit(0);
  }
  ROS_INFO_STREAM("data_loader: publishing frame " << imageIndex);
  string fileName = to_string(imageIndex++);

  string depth_fileName =
      string(6 - fileName.length(), '0') + fileName + "_left_depth.tiff";
  string depthFile = imageFolder + "/depth_image/" + depth_fileName;
  auto depthImage = cv::imread(depthFile, cv::ImreadModes::IMREAD_UNCHANGED);

  // for (int j = 0; j < 10; j ++){
  //         cout << "---" << depthImage.at<ushort>(0,j) << " ";
  //     }
  //     cout << endl;

  string image_fileName =
      string(6 - fileName.length(), '0') + fileName + "_left.png";
  string imageFile = imageFolder + "/image_left/" + image_fileName;
  // cout << "data_loader::load " << imageFile << endl;
  auto left = cv::imread(imageFile, cv::ImreadModes::IMREAD_COLOR);

  if (false) {
    // visualize depth image
    cv::Mat depth8U;
    depthImage.convertTo(depth8U, CV_8U, 1.0 / (50));
    cv::imshow("depth", depth8U);
    cv::waitKey(100);
  }
  depthImage.convertTo(depthImage, CV_16U, 1000);
  // odom trajectory
  auto &transform = odomPath.front();
  // path
  geometry_msgs::Point tmp;
  auto pos = get_posV2(transform);
  state.position = tf2::toMsg(pos, tmp);
  tf2::Quaternion quad = get_quadV2(transform);
  // quad.normalize();
  state.orientation = tf2::toMsg(quad);
  for (int i = 0; i < 9; ++i) {
    state.covariance[9 * i + i] = stateCov;
  }
  odomPath.pop();

  // camear info
  sensor_msgs::CameraInfo cameraInfoMsg;
  auto intrinsic = intrinsicG;
  cameraInfoMsg.K = boost::array<double, 9>{
      intrinsic(0, 0), intrinsic(0, 1), intrinsic(0, 2),
      intrinsic(1, 0), intrinsic(1, 1), intrinsic(1, 2),
      intrinsic(2, 0), intrinsic(2, 1), intrinsic(2, 2)};

  cameraInfoMsg.height = left.rows;
  cameraInfoMsg.width = left.cols;
  geometry_msgs::PoseStamped poseStamped;
  poseStamped.pose.orientation = state.orientation;
  poseStamped.pose.position = state.position;
  odomTrajectoryMsg.poses.push_back(poseStamped);

  // gt trajectory
  if (gtPath.empty()) {
    ROS_ERROR_STREAM("No gt for this frame");
  }
  auto &transformGT = gtPath.front();
  geometry_msgs::PoseStamped poseStampedGT;
  quad = get_quadV2(transformGT);
  poseStampedGT.pose.orientation = tf2::toMsg(quad);
  pos = get_posV2(transformGT);
  poseStampedGT.pose.position = tf2::toMsg(pos, tmp);
  gtTrajectoryMsg.poses.push_back(poseStampedGT);
  gtPath.pop();

  // publish
  std_msgs::Header header;
  header.seq = imageIndex;
  header.stamp = ros::Time::now();
  header.frame_id = worldFrame;
  pubLeftImage.publish(cv_bridge::CvImage(header, "bgr8", left).toImageMsg());
  pubDepth.publish(
      cv_bridge::CvImage(header, "16UC1", depthImage).toImageMsg());
  state.header = header;
  pubState.publish(state);
  pubCameraInfo.publish(cameraInfoMsg);
  odomTrajectoryMsg.header = header;
  pubOdomTrajectory.publish(odomTrajectoryMsg);
  gtTrajectoryMsg.header = header;
  pubTrajectoryGT.publish(gtTrajectoryMsg);

  // display the OpenCV Mat
  //   cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a
  //   window for display. cv::imshow("Display window", depth8U); // Show our
  //   image inside it. cv::waitKey(100); // Wait for a keystroke in the window
}

void handle_pose_with_cov(geometry_msgs::PoseWithCovarianceStamped::Ptr msg) {
  int sum = 0;
  for (int i = 0; i < 81; i++) {
    sum += abs(state.covariance[i]);
  }
  if (sum == 0) {
    // since cov for husky is 0, just insert a value on diagonal line
    for (int i = 0; i < 9; ++i) {
      state.covariance[9 * i + i] = stateCov;
    }
  } else {
    // insert 0 for velocity
    for (int i = 0; i < 9; i++) {
      for (int j = 0; j < 9; j++) {
        if (i >= 3 && i <= 5) {
          state.covariance[i * 9 + j] = 0;
        } else if (j >= 3 && j <= 5) {
          state.covariance[i * 9 + j] = 0;
        } else {
          int index = (i <= 6 ? i : i - 3) * 9 + (j <= 6 ? j : j - 3);
          state.covariance[i * 9 + j] = state.covariance[index];
        }
      }
    }
  }

  // construct the msg
  state.position = msg->pose.pose.position;
  state.orientation = msg->pose.pose.orientation;
  msg->header.frame_id = worldFrame;
  msg->header.stamp = ros::Time::now();
  state.header = msg->header;
  pubState.publish(state);

  // traj
  if ((ros::Time::now() - odomTrajectoryMsg.header.stamp).toSec() >= 0.2) {
    geometry_msgs::PoseStamped poseStamped;
    poseStamped.pose.orientation = msg->pose.pose.orientation;
    poseStamped.pose.position = msg->pose.pose.position;
    odomTrajectoryMsg.poses.push_back(poseStamped);
    odomTrajectoryMsg.header = state.header;
    pubOdomTrajectory.publish(odomTrajectoryMsg);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "state_publisher");
  ros::NodeHandle nh("~");
  // nh_ptr.reset(&nh);
  ros::Subscriber subTrajectory, subPose;
  string datasetType;
  nh.param<string>("dataset_type", datasetType, "tum_ros");
  nh.param<double>("state_covariance", stateCov, 0.01);
  if (datasetType == "tum_ros") {
    string pose_topic, trajectory_topic;
    nh.param<string>("pose_topic", pose_topic, "pose_in");
    nh.param<string>("trajectory_topic", trajectory_topic, "trajectory_in");

    subPose = nh.subscribe(pose_topic, 2000, &handle_pose);
    ROS_INFO("subscribing: %s", subPose.getTopic().c_str());

    subTrajectory = nh.subscribe(trajectory_topic, 2000, &handle_trajectory);
    ROS_INFO("subscribing: %s", subTrajectory.getTopic().c_str());
  } else if (datasetType == "kitti_raw") {
    // load params
    float baseline, fu, fv, cx, cy;
    Eigen::Matrix3f intrinsic;
    nh.param<int>("start_index", imageIndex, 0);
    nh.param<float>("warmup_duration", warmupDuration, 0);
    nh.param<float>("frame_duration", frameDuration, 0.16);
    nh.param<string>("image_folder", imageFolder, "");
    nh.param<string>("oxts_imu_folder", oxtsImuFolder, "");
    nh.param<string>("odom_file", odomFile, "");
    nh.param<string>("gt_file", gtFile, "");
    nh.param<string>("time_file", timeFile, "");
    nh.param<string>("world_frame", worldFrame, "world");
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
    pubCameraInfo = nh.advertise<sensor_msgs::CameraInfo>("camera_info", 10);
    pubImu = nh.advertise<sensor_msgs::Imu>("imu", 10);
    // read files
    read_time();
    read_traj(odomFile, odomPath);
    read_traj(gtFile, gtPath);
    read_oxts_imu(oxtsImuFolder, oxtsImuData);
    // publish images
    ros::Duration(warmupDuration).sleep();
    ros::Duration duration(frameDuration);
    durations.pop();
    timerImage = nh.createTimer(duration, publish_images_and_odom);
  } else if (datasetType == "pose_with_cov_to_state") {
    // load params
    string poseWithCovTopic;
    nh.param<string>("pose_with_cov_topic", poseWithCovTopic, "/pose");
    nh.param<string>("world_frame", worldFrame, "world");
    // publish trajectory
    subPose = nh.subscribe(poseWithCovTopic, 2000, &handle_pose_with_cov);
  } else if (datasetType == "tartanair") {
    float baseline, fu, fv, cx, cy;
    nh.param<int>("start_index", imageIndex, 0);
    nh.param<float>("warmup_duration", warmupDuration, 0);
    nh.param<float>("frame_duration", frameDuration, 0.16);
    nh.param<string>("image_folder", imageFolder, "");
    nh.param<string>("gt_file", gtFile, "");
    nh.param<string>("odom_file", odomFile, "");
    nh.param<string>("world_frame", worldFrame, "world");
    nh.param<float>("camera_baseline", baseline, 0.54);
    nh.param<float>("camera_fu", fu, 320.0);
    nh.param<float>("camera_fv", fv, 320.0);
    nh.param<float>("camera_cx", cx, 320.0);
    nh.param<float>("camera_cy", cy, 240.0);

    pubLeftImage = nh.advertise<sensor_msgs::Image>("color_left", 10);
    pubDepth = nh.advertise<sensor_msgs::Image>("depth", 10);
    pubCameraInfo = nh.advertise<sensor_msgs::CameraInfo>("camera_info", 10);
    intrinsicG << fu, 0, cx, 0, fv, cy, 0, 0, 1;
    read_traj(gtFile, gtPath);
    read_traj(odomFile, odomPath);
    ros::Duration(warmupDuration).sleep();
    timerImage = nh.createTimer(ros::Duration(frameDuration),
                                publish_images_and_odom_sc);
  }
  pubState = nh.advertise<inekf_msgs::State>("state", 10);
  pubTrajectoryGT = nh.advertise<nav_msgs::Path>("gt_trajectory", 10);
  pubOdomTrajectory = nh.advertise<nav_msgs::Path>("odom_trajectory", 10);

  ros::spin();
}