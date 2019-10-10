/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */

#include "external.hpp"
#include <iostream>

namespace rangenet {
namespace external {

void print_flags(void) {
#ifdef TENSORRT_FOUND
  std::cout << "[TENSORRT_FOUND] Defined" << std::endl;
#else
  std::cerr << "[TENSORRT_FOUND] NOT defined" << std::endl;
#endif
}
}  // namespace external
}  // namespace rangenet
