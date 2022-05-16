#include <ros/ros.h>
#include <string>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <inekf_msgs/State.h>
#include <nav_msgs/Path.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace std;

ros::Publisher pubState, pubTrajectoryGT;
inekf_msgs::State state;

bool trajectoryInitialized = false;

void handle_pose(geometry_msgs::PoseStamped::Ptr msg)
{
    // pose to inekf state
    state.header = msg->header;
    state.position = msg->pose.position;
    state.orientation = msg->pose.orientation;
    pubState.publish(state);

    // align backend trajectory to gt
    static tf2_ros::Buffer tfBuffer;
    static tf2_ros::TransformListener tfListener(tfBuffer);
    static tf2_ros::TransformBroadcaster br;
    static geometry_msgs::TransformStamped transformStampedPub;
    if (!trajectoryInitialized)
    {
        try
        {
            transformStampedPub = tfBuffer.lookupTransform("world", "kinect", ros::Time(0));
            transformStampedPub.child_frame_id = "trajectory";
        }
        catch (tf2::TransformException &ex)
        {
            return;
        }
    }
    transformStampedPub.header = msg->header;
    br.sendTransform(transformStampedPub);

    // publish gt trajectory
    static nav_msgs::Path trajectoryMsg;
    geometry_msgs::PoseStamped poseStamped;
    static geometry_msgs::TransformStamped transformStamped;
    try
    {
        transformStamped = tfBuffer.lookupTransform("world", "kinect", ros::Time(0));
    }
    catch (tf2::TransformException &ex)
    {
        return;
    }
    poseStamped.pose.orientation = transformStamped.transform.rotation;
    poseStamped.pose.position.x = transformStamped.transform.translation.x;
    poseStamped.pose.position.y = transformStamped.transform.translation.y;
    poseStamped.pose.position.z = transformStamped.transform.translation.z;
    trajectoryMsg.poses.push_back(poseStamped);
    trajectoryMsg.header = msg->header;
    pubTrajectoryGT.publish(trajectoryMsg);
}

void handle_trajectory(nav_msgs::Path::Ptr msg)
{
    trajectoryInitialized = true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "state_publisher");
    ros::NodeHandle nh("~");

    string pose_topic, trajectory_topic;
    nh.param<string>("pose_topic", pose_topic, "pose_in");
    nh.param<string>("trajectory_topic", trajectory_topic, "trajectory_in");

    ros::Subscriber subPose = nh.subscribe(pose_topic, 2000, &handle_pose);
    ROS_INFO("subscribing: %s", subPose.getTopic().c_str());

    ros::Subscriber subTrajectory = nh.subscribe(trajectory_topic, 2000, &handle_trajectory);
    ROS_INFO("subscribing: %s", subTrajectory.getTopic().c_str());

    pubState = nh.advertise<inekf_msgs::State>("state", 10);
    pubTrajectoryGT = nh.advertise<nav_msgs::Path>("gt_trajectory", 10);

    ros::spin();
}