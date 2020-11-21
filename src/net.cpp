/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#include "net.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace rangenet {
namespace segmentation {

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the inference model directory
 */
Net::Net(const std::string& model_path)
    : _model_path(model_path), _verbose(false) {
  // set default verbosity level
  verbosity(_verbose);

  // Try to get the config file as well
  std::string arch_cfg_path = _model_path + "/arch_cfg.yaml";
  try {
    arch_cfg = YAML::LoadFile(arch_cfg_path);
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Can't open cfg.yaml from " + arch_cfg_path);
  }

  // Assign fov_up and fov_down from arch_cfg
  _fov_up = arch_cfg["dataset"]["sensor"]["fov_up"].as
  <double>();
  _fov_down = arch_cfg["dataset"]["sensor"]["fov_down"].as
  <double>();

  std::string data_cfg_path = _model_path + "/data_cfg.yaml";
  try {
    data_cfg = YAML::LoadFile(data_cfg_path);
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Can't open cfg.yaml from " + data_cfg_path);
  }

  // Get label dictionary from yaml cfg
  YAML::Node color_map;
  try {
    color_map = data_cfg["color_map"];
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one the label dictionary from cfg in " + data_cfg_path
              << std::endl;
    throw ex;
  }

  // Generate string map from xentropy indexes (that we'll get from argmax)
  YAML::const_iterator it;

  for (it = color_map.begin(); it != color_map.end(); ++it) {
    // Get label and key
    int key = it->first.as<int>();  // <- key
    Net::color color = std::make_tuple(
        static_cast<u_char>(color_map[key][0].as<unsigned int>()),
        static_cast<u_char>(color_map[key][1].as<unsigned int>()),
        static_cast<u_char>(color_map[key][2].as<unsigned int>()));
    _color_map[key] = color;
  }

  // Get learning class labels from yaml cfg
  YAML::Node learning_class;
  try {
    learning_class = data_cfg["learning_map_inv"];
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one the label dictionary from cfg in " + data_cfg_path
              << std::endl;
    throw ex;
  }

  // get the number of classes
  _n_classes = learning_class.size();

  // remapping the colormap lookup table
  _lable_map.resize(_n_classes);
  for (it = learning_class.begin(); it != learning_class.end(); ++it) {
    int key = it->first.as<int>();  // <- key
    _argmax_to_rgb[key] = _color_map[learning_class[key].as<unsigned int>()];
    _lable_map[key] = learning_class[key].as<unsigned int>();
  }

  // get image size
  _img_h = arch_cfg["dataset"]["sensor"]["img_prop"]["height"].as<int>();
  _img_w = arch_cfg["dataset"]["sensor"]["img_prop"]["width"].as<int>();
  _img_d = 5; // range, x, y, z, remission

  // get normalization parameters
  YAML::Node img_means, img_stds;
  try {
    img_means = arch_cfg["dataset"]["sensor"]["img_means"];
    img_stds = arch_cfg["dataset"]["sensor"]["img_stds"];
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one the mean or std dictionary from cfg"
              << std::endl;
    throw ex;
  }
  // fill in means from yaml node
  for (it = img_means.begin(); it != img_means.end(); ++it) {
    // Get value
    float mean = it->as<float>();
    // Put in indexing vector
    _img_means.push_back(mean);
  }
  // fill in stds from yaml node
  for (it = img_stds.begin(); it != img_stds.end(); ++it) {
    // Get value
    float std = it->as<float>();
    // Put in indexing vector
    _img_stds.push_back(std);
  }
}

/**
 * @brief      Get raw point clouds
 *
 * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
 *
 * @return     cv format points
 */
std::vector<cv::Vec3f> Net::getPoints(const std::vector<float>& scan, const uint32_t &num_points) {
  std::vector<cv::Vec3f> points;
  points.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    points[i] = cv::Vec3f(scan[4 * i], scan[4 * i + 1], scan[4 * i +2]);
  }
  return points;
}

/**
 * @brief      Convert mask to color using dictionary as lut
 *
 * @param[in]  semantic_scan, The mask from argmax; num_points, the number of points in this scan.
 *
 * @return     the colored segmentation mask :)
 */
std::vector<cv::Vec3b> Net::getLabels(const std::vector<std::vector<float>>& semantic_scan, const uint32_t &num_points) {
  std::vector<cv::Vec3b> labels;
  std::vector<float> labels_prob;
  labels.resize(num_points);
  labels_prob.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    labels_prob[i] = 0;
    for (int32_t j = 0; j < _n_classes; ++j)
    {
      if (labels_prob[i] <= semantic_scan[i][j])
      {
        labels[i] = cv::Vec3b(std::get<0>(_argmax_to_rgb[j]),
                              std::get<1>(_argmax_to_rgb[j]),
                              std::get<2>(_argmax_to_rgb[j]));
        labels_prob[i] = semantic_scan[i][j];
      }
    }
  }
  return labels;
}

}  // namespace segmentation
}  // namespace rangenet
