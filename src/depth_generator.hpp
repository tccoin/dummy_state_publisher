#pragma once

#include <eigen3/Eigen/Core>
#include <opencv2/core/core.hpp>

#include "libelas/elas.h"

using namespace std;

enum TraceStatus { GOOD = 0, OOB, OUTLIER };

class DepthGenerator {
 public:
  float baseline;
  Eigen::Matrix3f intrinsic;
  shared_ptr<Elas> elas;

  DepthGenerator() {
    baseline = 0.53790448813;
    float fu = 707.0912, fv = 707.0912, cx = 601.8873, cy = 183.1104;
    // nh.param<float>("camera_baseline", baseLine, 0.54);
    // nh.param<float>("camera_fu", baseLine, 707.0912);
    // nh.param<float>("camera_fv", baseLine, 601.8873);
    // nh.param<float>("camera_cx", baseLine, 707.0912);
    // nh.param<float>("camera_cy", baseLine, 183.1104);
    intrinsic << fu, 0, cx, 0, fv, cy, 0, 0, 1;

    // elas
    Elas::parameters p;
    p.postprocess_only_left = true;
    elas.reset(new Elas(p));
  }

  TraceStatus pt_depth_from_disparity(const cv::Mat& left,
                                      const std::vector<float>& disparity,
                                      int u, int v, Eigen::Vector3f& result);

  void disparity(const cv::Mat& left_in, const cv::Mat& right_in,
                 std::vector<float>& output_left_disparity);
};