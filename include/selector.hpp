/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#pragma once

// standard stuff
#include <algorithm>
#include <iostream>
#include <string>

// selective network library (conditional build)
#include "external.hpp"  //this one contains the flags for external lib build
#ifdef TENSORRT_FOUND
#include "netTensorRT.hpp"
#endif

// Only to be used with segmentation
namespace rangenet {
namespace segmentation {

/**
 * @brief Makes a network with the desired backend, checking that it exists,
 *        it is implemented, and that it was compiled.
 *
 * @param backend "tensorrt", only tensorrt implemented
 * @return std::unique_ptr<Net>
 */
std::unique_ptr<Net> make_net(const std::string& path,
                              const std::string& backend);

}  // namespace segmentation
}  // namespace rangenet
