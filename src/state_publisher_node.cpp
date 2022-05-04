#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <inekf_msgs/State.h>
#include <string>

using namespace std;

ros::Publisher pubState;

void handle_pose(geometry_msgs::PoseStamped::Ptr msg)
{
    inekf_msgs::State state;
    state.header = msg->header;
    state.position = msg->pose.position;
    state.orientation = msg->pose.orientation;
    pubState.publish(state);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "state_publisher");
    ros::NodeHandle nh("~");

    ros::Subscriber subPose = nh.subscribe("pose_in", 2000, &handle_pose);
    ROS_INFO("subscribing: %s", subPose.getTopic().c_str());

    pubState = nh.advertise<inekf_msgs::State>("state", 10);

    ros::spin();
}