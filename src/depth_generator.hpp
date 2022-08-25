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

  DepthGenerator(float baseline_, Eigen::Matrix3f intrinsic_)
      : baseline(baseline_), intrinsic(intrinsic_) {
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
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};