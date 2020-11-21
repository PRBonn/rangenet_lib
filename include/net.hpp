/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#pragma once

// standard stuff
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// opencv
#include <opencv2/core/core.hpp>

// yamlcpp
#include "yaml-cpp/yaml.h"

namespace rangenet {
namespace segmentation {

/**
 * @brief      Class for segmentation network inference.
 */
class Net {
 public:
  typedef std::tuple< u_char, u_char, u_char> color;
  /**
   * @brief      Constructs the object.
   *
   * @param[in]  model_path  The model path for the inference model directory
   */
  Net(const std::string& model_path);

  /**
   * @brief      Destroys the object.
   */
  virtual ~Net(){};

  /**
   * @brief      Infer logits from LiDAR scan
   *
   * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
   *
   * @return     Semantic estimates with probabilities over all classes (_n_classes, _img_h, _img_w)
   */
  virtual std::vector<std::vector<float>> infer(const std::vector<float>& scan, const uint32_t &num_points) = 0;

  /**
   * @brief      Get raw point clouds
   *
   * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
   *
   * @return     cv format points
   */
  std::vector<cv::Vec3f> getPoints(const std::vector<float> &scan, const uint32_t& num_points);

  /**
   * @brief      Convert mask to color using dictionary as lut
   *
   * @param[in]  semantic_scan, The mask from argmax; num_points, the number of points in this scan.
   *
   * @return     the colored segmentation mask :)
   */
  std::vector<cv::Vec3b> getLabels(const std::vector<std::vector<float> > &semantic_scan, const uint32_t& num_points);


  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  void verbosity(const bool verbose) { _verbose = verbose; }

  /**
   * @brief      Get the label map
   *
   * @return     the learning label mapping
   */
  std::vector<int> getLabelMap() { return _lable_map;}

  /**
   * @brief      Get the color map
   *
   * @return     the color map
   */
  std::map<uint32_t, color> getColorMap() { return _color_map;}

 protected:
  // general
  std::string _model_path;  // Where to get model weights and cfg
  bool _verbose;            // verbose mode?

  // image properties
  int _img_h, _img_w, _img_d;  // height, width, and depth for inference
  std::vector<float> _img_means, _img_stds; // mean and std per channel
  // problem properties
  int32_t _n_classes;  // number of classes to differ from
  // sensor properties
  double _fov_up, _fov_down; // field of view up and down in radians

  // config
  YAML::Node data_cfg;  // yaml nodes with configuration from training
  YAML::Node arch_cfg;  // yaml nodes with configuration from training

  std::vector<int> _lable_map;
  std::map<uint32_t, color> _color_map;
  std::map<uint32_t, color> _argmax_to_rgb;  // for color conversion
};

}  // namespace segmentation
}  // namespace rangenet
