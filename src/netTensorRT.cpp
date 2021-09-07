/* Copyright (c) 2019 Xieyuanli Chen, Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */

#include "netTensorRT.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>

namespace rangenet {
namespace segmentation {

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the inference model directory
 *                         containing the "model.trt" file and the cfg
 */
NetTensorRT::NetTensorRT(const std::string& model_path)
    : Net(model_path), _engine(0), _context(0) {
  // set default verbosity level
  verbosity(_verbose);

  // Try to open the model
  std::cout << "Trying to open model" << std::endl;

  // generate trt path form model path
  std::string engine_path = model_path + "/model.trt";

  // try to deserialize the engine
  try {
    deserializeEngine(engine_path);
  } catch (std::exception e) {
    std::cout << "Could not deserialize TensorRT engine. " << std::endl
              << "Generating from sratch... This may take a while..."
              << std::endl;

    // destroy crap from engine
    if (_engine) _engine->destroy();

  } catch (...) {
    throw std::runtime_error("Unknown TensorRT exception. Giving up.");
  }

  // if there is no engine, try to generate one from onnx
  if (!_engine) {
    // generate path
    std::string onnx_path = model_path + "/model.onnx";
    // generate engine
    generateEngine(onnx_path);
    // save engine
    serializeEngine(engine_path);
  }

  // prepare buffers for io :)
  prepareBuffer();

  CUDA_CHECK(cudaStreamCreate(&_cudaStream));

}  // namespace segmentation

/**
 * @brief      Destroys the object.
 */
NetTensorRT::~NetTensorRT() {
  if (_verbose) {
    std::cout << "start to destroy the process." << std::endl;
  }
  // free cuda buffers
  int n_bindings = _engine->getNbBindings();
  for (int i = 0; i < n_bindings; i++) {
    CUDA_CHECK(cudaFree(_deviceBuffers[i]));
  }

  if (_verbose) {
    std::cout << "cuda buffers released." << std::endl;
  }

  // free cuda pinned mem
  for (auto& buffer : _hostBuffers) CUDA_CHECK(cudaFreeHost(buffer));

  if (_verbose) {
    std::cout << "cuda pinned mem released." << std::endl;
  }

  // destroy cuda stream
  CUDA_CHECK(cudaStreamDestroy(_cudaStream));

  if (_verbose) {
    std::cout << "cuda stream destroyed." << std::endl;
  }

  // destroy the execution context
  if (_context) {
    _context->destroy();
  }

  if (_verbose) {
    std::cout << "execution context destroyed." << std::endl;
  }

  // destroy the engine
  if (_engine) {
    _engine->destroy();
  }

  if (_verbose) {
    std::cout << "engine destroyed." << std::endl;
  }
}

/**
 * @brief      Project a pointcloud into a spherical projection image.projection.
 *
 * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
 *
 * @return     Projected LiDAR scans, with size of (_img_h * _img_w, _img_d)
 */
std::vector<std::vector<float>> NetTensorRT::doProjection(const std::vector<float>& scan, const uint32_t& num_points){
  float fov_up = _fov_up / 180.0 * M_PI;    // field of view up in radians
  float fov_down = _fov_down / 180.0 * M_PI;  // field of view down in radians
  float fov = std::abs(fov_down) + std::abs(fov_up); // get field of view total in radians

  std::vector<float> ranges;
  std::vector<float> xs;
  std::vector<float> ys;
  std::vector<float> zs;
  std::vector<float> intensitys;

  std::vector<float> proj_xs_tmp;
  std::vector<float> proj_ys_tmp;

  for (uint32_t i = 0; i < num_points; i++) {
    float x = scan[4 * i];
    float y = scan[4 * i + 1];
    float z = scan[4 * i + 2];
    float intensity = scan[4 * i + 3];
    float range = std::sqrt(x*x+y*y+z*z);
    ranges.push_back(range);
    xs.push_back(x);
    ys.push_back(y);
    zs.push_back(z);
    intensitys.push_back(intensity);

    // get angles
    float yaw = -std::atan2(y, x);
    float pitch = std::asin(z / range);

    // get projections in image coords
    float proj_x = 0.5 * (yaw / M_PI + 1.0); // in [0.0, 1.0]
    float proj_y = 1.0 - (pitch + std::abs(fov_down)) / fov; // in [0.0, 1.0]

    // scale to image size using angular resolution
    proj_x *= _img_w; // in [0.0, W]
    proj_y *= _img_h; // in [0.0, H]

    // round and clamp for use as index
    proj_x = std::floor(proj_x);
    proj_x = std::min(_img_w - 1.0f, proj_x);
    proj_x = std::max(0.0f, proj_x); // in [0,W-1]
    proj_xs_tmp.push_back(proj_x);

    proj_y = std::floor(proj_y);
    proj_y = std::min(_img_h - 1.0f, proj_y);
    proj_y = std::max(0.0f, proj_y); // in [0,H-1]
    proj_ys_tmp.push_back(proj_y);
  }

  // stope a copy in original order
  proj_xs = proj_xs_tmp;
  proj_ys = proj_ys_tmp;

  // order in decreasing depth
  std::vector<size_t> orders = sort_indexes(ranges);
  std::vector<float> sorted_proj_xs;
  std::vector<float> sorted_proj_ys;
  std::vector<std::vector<float>> inputs;

  for (size_t idx : orders){
    sorted_proj_xs.push_back(proj_xs[idx]);
    sorted_proj_ys.push_back(proj_ys[idx]);
    std::vector<float> input = {ranges[idx], xs[idx], ys[idx], zs[idx], intensitys[idx]};
    inputs.push_back(input);
  }

  // assing to images
  std::vector<std::vector<float>> range_image(_img_w * _img_h);

  // zero initialize
  for (uint32_t i = 0; i < range_image.size(); ++i) {
      range_image[i] = invalid_input;
  }

  for (uint32_t i = 0; i < inputs.size(); ++i) {
    range_image[int(sorted_proj_ys[i] * _img_w + sorted_proj_xs[i])] = inputs[i];
  }

  return range_image;
}

/**
 * @brief      Infer logits from LiDAR scan
 *
 * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
 *
 * @return     Semantic estimates with probabilities over all classes (_n_classes, _img_h, _img_w)
 */
std::vector<std::vector<float>> NetTensorRT::infer(const std::vector<float>& scan, const uint32_t& num_points) {
  // check if engine is valid
  if (!_engine) {
    throw std::runtime_error("Invaild engine on inference.");
  }

  // start inference
  if (_verbose) {
    tic();
    std::cout << "Inferring with TensorRT" << std::endl;
    tic();
  }

  // project point clouds into range image
  std::vector<std::vector<float>> projected_data = doProjection(scan, num_points);


  if (_verbose) {
    std::cout << "Time for projection: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // put in buffer using position
  int channel_offset = _img_h * _img_w;

  bool all_zeros = false;
  std::vector<int> invalid_idxs;

  for (uint32_t pixel_id = 0; pixel_id < projected_data.size(); pixel_id++){
    // check if the pixel is invalid
    all_zeros = std::all_of(projected_data[pixel_id].begin(), projected_data[pixel_id].end(), [](int i) { return i==0.0f; });
    if (all_zeros) {
      invalid_idxs.push_back(pixel_id);
    }
    for (int i = 0; i < _img_d; i++) {
      // normalize the data
      if (!all_zeros) {
        projected_data[pixel_id][i] = (projected_data[pixel_id][i] - this->_img_means[i]) / this->_img_stds[i];
      }

      int buffer_idx = channel_offset * i + pixel_id;
      ((float*)_hostBuffers[_inBindIdx])[buffer_idx] = projected_data[pixel_id][i];
    }
  }

  // clock now
  if (_verbose) {
    std::cout << "Time for preprocessing: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // execute inference
  CUDA_CHECK(
      cudaMemcpyAsync(_deviceBuffers[_inBindIdx], _hostBuffers[_inBindIdx],
                      getBufferSize(_engine->getBindingDimensions(_inBindIdx),
                                    _engine->getBindingDataType(_inBindIdx)),
                      cudaMemcpyHostToDevice, _cudaStream));
  if (_verbose) {
    CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    std::cout << "Time for copy in: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }


  _context->enqueue(1, &_deviceBuffers[_inBindIdx], _cudaStream, nullptr);

  if (_verbose) {
    CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    std::cout << "Time for inferring: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  CUDA_CHECK(
      cudaMemcpyAsync(_hostBuffers[_outBindIdx], _deviceBuffers[_outBindIdx],
                      getBufferSize(_engine->getBindingDimensions(_outBindIdx),
                                    _engine->getBindingDataType(_outBindIdx)),
                      cudaMemcpyDeviceToHost, _cudaStream));
  CUDA_CHECK(cudaStreamSynchronize(_cudaStream));

  if (_verbose) {
    std::cout << "Time for copy back: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // take the data out
  std::vector<std::vector<float>> range_image(channel_offset);
  for (int pixel_id = 0; pixel_id < channel_offset; pixel_id++){
    for (int i = 0; i < _n_classes; i++) {
      int buffer_idx = channel_offset * i + pixel_id;
      range_image[pixel_id].push_back(((float*)_hostBuffers[_outBindIdx])[buffer_idx]);
    }
  }

  if (_verbose) {
    std::cout << "Time for taking the data out: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // set invalid pixels
  for (int idx : invalid_idxs) {
    range_image[idx] = invalid_output;
  }

  // unprojection, labelling raw point clouds
  std::vector<std::vector<float>> semantic_scan;
  for (uint32_t i = 0 ; i < num_points; i++) {
    semantic_scan.push_back(range_image[proj_ys[i] * _img_w + proj_xs[i]]);
  }

  if (_verbose) {
    std::cout << "Time for unprojection: "
          << toc() * 1000
          << "ms" << std::endl;
    std::cout << "Time for the whole: "
              << toc() * 1000
              << "ms" << std::endl;
  }

  return semantic_scan;
}

/**
 * @brief      Set verbosity level for backend execution
 *
 * @param[in]  verbose  True is max verbosity, False is no verbosity.
 *
 * @return     Exit code.
 */
void NetTensorRT::verbosity(const bool verbose) {
  std::cout << "Setting verbosity to: " << (verbose ? "true" : "false")
            << std::endl;

  // call parent class verbosity
  this->Net::verbosity(verbose);

  // set verbosity for tensorRT logger
  _gLogger.set_verbosity(verbose);
}

/**
 * @brief Get the Buffer Size object
 *
 * @param d dimension
 * @param t data type
 * @return int size of data
 */
int NetTensorRT::getBufferSize(Dims d, DataType t) {
  int size = 1;
  for (int i = 0; i < d.nbDims; i++) size *= d.d[i];

  switch (t) {
    case DataType::kINT32:
      return size * 4;
    case DataType::kFLOAT:
      return size * 4;
    case DataType::kHALF:
      return size * 2;
    case DataType::kINT8:
      return size * 1;
    default:
      throw std::runtime_error("Data type not handled");
  }
  return 0;
}

/**
 * @brief Deserialize an engine that comes from a previous run
 *
 * @param engine_path
 */
void NetTensorRT::deserializeEngine(const std::string& engine_path) {
  // feedback to user where I am
  std::cout << "Trying to deserialize previously stored: " << engine_path
            << std::endl;

  // open model if it exists, otherwise complain
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);
  std::ifstream file_ifstream(engine_path.c_str());
  if (file_ifstream) {
    std::cout << "Successfully found TensorRT engine file " << engine_path
              << std::endl;
  } else {
    throw std::runtime_error("TensorRT engine file not found" + engine_path);
  }

  // create inference runtime
  IRuntime* infer = createInferRuntime(_gLogger);
  if (infer) {
    std::cout << "Successfully created inference runtime" << std::endl;
  } else {
    throw std::runtime_error("Couldn't created inference runtime.");
  }

// if using DLA, set the desired core before deserialization occurs
#if NV_TENSORRT_MAJOR >= 5 &&                             \
    !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && \
      NV_TENSORRT_PATCH == 0)
  if (DEVICE_DLA_0) {
    infer->setDLACore(0);
    std::cout << "Successfully selected DLA core 0." << std::endl;
  } else if (DEVICE_DLA_1) {
    infer->setDLACore(1);
    std::cout << "Successfully selected DLA core 1." << std::endl;
  } else {
    std::cout << "No DLA selected." << std::endl;
  }
#endif

  // read file
  gieModelStream << file_ifstream.rdbuf();
  file_ifstream.close();
  // read the stringstream into a memory buffer and pass that to TRT.
  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);
  void* modelMem = malloc(modelSize);
  if (modelMem) {
    std::cout << "Successfully allocated " << modelSize << " for model."
              << std::endl;
  } else {
    throw std::runtime_error("failed to allocate " + std::to_string(modelSize) +
                             " bytes to deserialize model");
  }
  gieModelStream.read((char*)modelMem, modelSize);
  std::cout << "Successfully read " << modelSize << " to modelmem."
            << std::endl;

  // because I use onnx-tensorRT i have to use their plugin factory
  nvonnxparser::IPluginFactory* plug_fact =
      nvonnxparser::createPluginFactory(_gLogger);

  // Now deserialize
  _engine = infer->deserializeCudaEngine(modelMem, modelSize, plug_fact);

  free(modelMem);
  if (_engine) {
    std::cerr << "Created engine!" << std::endl;
  } else {
    throw std::runtime_error("Device failed to create CUDA engine");
  }

  std::cout << "Successfully deserialized Engine from trt file" << std::endl;
}

/**
 * @brief Serialize an engine that we generated in this run
 *
 * @param engine_path
 */
void NetTensorRT::serializeEngine(const std::string& engine_path) {
  // feedback to user where I am
  std::cout << "Trying to serialize engine and save to : " << engine_path
            << " for next run" << std::endl;

  // do only if engine is healthy
  if (_engine) {
    // do the serialization
    IHostMemory* engine_plan = _engine->serialize();
    // Try to save engine for future uses.
    std::ofstream stream(engine_path.c_str(), std::ofstream::binary);
    if (stream)
      stream.write(static_cast<char*>(engine_plan->data()),
                   engine_plan->size());
  }
}

/**
 * @brief Generate an engine from ONNX model
 *
 * @param onnx_path path to onnx file
 */
void NetTensorRT::generateEngine(const std::string& onnx_path) {
  // feedback to user where I am
  std::cout << "Trying to generate trt engine from : " << onnx_path
            << std::endl;

  // create inference builder
  IBuilder* builder = createInferBuilder(_gLogger);

  // set optimization parameters here
  // CAN I DO HALF PRECISION (and report to user)
  std::cout << "Platform ";
  if (builder->platformHasFastFp16()) {
    std::cout << "HAS ";
    builder->setFp16Mode(true);
  } else {
    std::cout << "DOESN'T HAVE ";
    builder->setFp16Mode(false);
  }
  std::cout << "fp16 support." << std::endl;
  // BATCH SIZE IS ALWAYS ONE
  builder->setMaxBatchSize(1);

// if using DLA, set the desired core before deserialization occurs
#if NV_TENSORRT_MAJOR >= 5 &&                             \
    !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && \
      NV_TENSORRT_PATCH == 0)
  if (DEVICE_DLA_0 || DEVICE_DLA_1) {
    builder->setDefaultDeviceType(DeviceType::kDLA);
    builder->allowGPUFallback(true);
    if (DEVICE_DLA_0) {
      std::cout << "Successfully selected DLA core 0." << std::endl;
      builder->setDLACore(0);
    } else if (DEVICE_DLA_0) {
      std::cout << "Successfully selected DLA core 1." << std::endl;
      builder->setDLACore(1);
    }
  } else {
    std::cout << "No DLA selected." << std::endl;
  }
#endif

  // create a network builder
  INetworkDefinition* network = builder->createNetwork();

  // generate a parser to get weights from onnx file
  nvonnxparser::IParser* parser =
      nvonnxparser::createParser(*network, _gLogger);

  // finally get from file
  if (!parser->parseFromFile(onnx_path.c_str(),
                             static_cast<int>(ILogger::Severity::kVERBOSE))) {
    throw std::runtime_error("ERROR: could not parse input ONNX.");
  } else {
    std::cout << "Success picking up ONNX model" << std::endl;
  }

  // put in engine
  // iterate until I find a size that fits
  for (unsigned long ws_size = MAX_WORKSPACE_SIZE;
       ws_size >= MIN_WORKSPACE_SIZE; ws_size /= 2) {
    // set size
    builder->setMaxWorkspaceSize(ws_size);

    // try to build
    _engine = builder->buildCudaEngine(*network);
    if (!_engine) {
      std::cerr << "Failure creating engine from ONNX model" << std::endl
                << "Current trial size is " << ws_size << std::endl;
      continue;
    } else {
      std::cout << "Success creating engine from ONNX model" << std::endl
                << "Final size is " << ws_size << std::endl;
      break;
    }
  }

  // final check
  if (!_engine) {
    throw std::runtime_error("ERROR: could not create engine from ONNX.");
  } else {
    std::cout << "Success creating engine from ONNX model" << std::endl;
  }
}

/**
 * @brief Prepare io buffers for inference with engine
 */
void NetTensorRT::prepareBuffer() {
  // check if engine is ok
  if (!_engine) {
    throw std::runtime_error(
        "Invalid engine. Please remember to create engine first.");
  }

  // get execution context from engine
  _context = _engine->createExecutionContext();
  if (!_context) {
    throw std::runtime_error("Invalid execution context. Can't infer.");
  }

  int n_bindings = _engine->getNbBindings();
  if (n_bindings != 2) {
    throw std::runtime_error("Invalid number of bindings: " +
                             std::to_string(n_bindings));
  }

  // clear buffers and reserve memory
  _deviceBuffers.clear();
  _deviceBuffers.reserve(n_bindings);
  _hostBuffers.clear();
  _hostBuffers.reserve(n_bindings);

  // allocate memory
  for (int i = 0; i < n_bindings; i++) {
    nvinfer1::Dims dims = _engine->getBindingDimensions(i);
    nvinfer1::DataType dtype = _engine->getBindingDataType(i);
    CUDA_CHECK(cudaMalloc(&_deviceBuffers[i],
                          getBufferSize(_engine->getBindingDimensions(i),
                                        _engine->getBindingDataType(i))));

    CUDA_CHECK(cudaMallocHost(&_hostBuffers[i],
                              getBufferSize(_engine->getBindingDimensions(i),
                                            _engine->getBindingDataType(i))));

    if (_engine->bindingIsInput(i))
      _inBindIdx = i;
    else
      _outBindIdx = i;

    std::cout << "Binding: " << i << ", type: " << (int)dtype << std::endl;
    for (int d = 0; d < dims.nbDims; d++) {
      std::cout << "[Dim " << dims.d[d] << "]";
    }
    std::cout << std::endl;
  }

  // exit
  std::cout << "Successfully create binding buffer" << std::endl;
}

}  // namespace segmentation
}  // namespace rangenet
