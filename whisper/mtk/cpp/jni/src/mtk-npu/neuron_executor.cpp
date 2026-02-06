/**
 * Neuron Executor Implementation
 */

#include "neuron_executor.h"
#include <iostream>
#include <cstring>
#include <dlfcn.h>
#include <sstream>

#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define LOG_WARNING(msg) std::cout << "[WARNING] " << msg << std::endl

// Helper function to join vector of integers into string
static std::string JoinString(const std::vector<uint32_t>& vec, const std::string& delim) {
    if (vec.empty()) return "";
    std::ostringstream oss;
    oss << vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
        oss << delim << vec[i];
    }
    return oss.str();
}

// NeuronOperandType structure (compatible with MTK Neuron API)
struct NeuronOperandType {
    int32_t type;
    float scale;
    int32_t zeroPoint;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
};

NeuronExecutor::NeuronExecutor(
    const std::string& model_path,
    const std::vector<std::vector<uint32_t>>& input_shapes,
    const std::vector<std::vector<uint32_t>>& output_shapes,
    const std::string& name)
    : model_path_(model_path),
      name_(name),
      input_shapes_(input_shapes),
      output_shapes_(output_shapes) {
}

NeuronExecutor::~NeuronExecutor() {
    // Cleanup in reverse order
    if (execution_) {
        if (fn_execution_free_) {
            fn_execution_free_(execution_);
        }
        execution_ = nullptr;
    }

    if (compilation_) {
        if (fn_compilation_free_) {
            fn_compilation_free_(compilation_);
        }
        compilation_ = nullptr;
    }

    if (model_) {
        if (fn_model_free_) {
            fn_model_free_(model_);
        }
        model_ = nullptr;
    }

    // Unmap DLA file
    if (dla_buffer_ && dla_size_ > 0) {
        munmap(dla_buffer_, dla_size_);
        dla_buffer_ = nullptr;
        dla_size_ = 0;
    }

    if (dla_fd_ >= 0) {
        close(dla_fd_);
        dla_fd_ = -1;
    }

    // Close library
    if (neuron_lib_) {
        dlclose(neuron_lib_);
        neuron_lib_ = nullptr;
    }
}

bool NeuronExecutor::LoadNeuronAPI() {
    LOG_INFO("Loading Neuron API library...");

    // Try to load libneuronusdk_adapter first (new API)
    neuron_lib_ = dlopen("libneuronusdk_adapter.mtk.so", RTLD_LAZY);
    if (!neuron_lib_) {
        // Fallback to libneuron_adapter
        neuron_lib_ = dlopen("libneuron_adapter.so", RTLD_LAZY);
    }

    if (!neuron_lib_) {
        LOG_ERROR("Failed to load Neuron adapter library: " << dlerror());
        return false;
    }

    // Load Model functions
    fn_model_create_ = (FnNeuronModel_create)dlsym(neuron_lib_, "NeuronModel_create");
    fn_model_free_ = (FnNeuronModel_free)dlsym(neuron_lib_, "NeuronModel_free");
    fn_add_operand_ = (FnNeuronModel_addOperand)dlsym(neuron_lib_, "NeuronModel_addOperand");
    fn_get_ext_operand_type_ = (FnNeuronModel_getExtensionOperandType)dlsym(
        neuron_lib_, "NeuronModel_getExtensionOperandType");
    fn_get_ext_operation_type_ = (FnNeuronModel_getExtensionOperationType)dlsym(
        neuron_lib_, "NeuronModel_getExtensionOperationType");
    fn_set_operand_value_ = (FnNeuronModel_setOperandValue)dlsym(
        neuron_lib_, "NeuronModel_setOperandValue");
    fn_add_operation_ = (FnNeuronModel_addOperation)dlsym(
        neuron_lib_, "NeuronModel_addOperation");
    fn_identify_inputs_outputs_ = (FnNeuronModel_identifyInputsAndOutputs)dlsym(
        neuron_lib_, "NeuronModel_identifyInputsAndOutputs");
    fn_model_finish_ = (FnNeuronModel_finish)dlsym(neuron_lib_, "NeuronModel_finish");

    // Load Compilation functions
    fn_compilation_create_ = (FnNeuronCompilation_createWithOptions)dlsym(
        neuron_lib_, "NeuronCompilation_createWithOptions");
    fn_compilation_free_ = (FnNeuronCompilation_free)dlsym(
        neuron_lib_, "NeuronCompilation_free");
    fn_set_priority_ = (FnNeuronCompilation_setPriority)dlsym(
        neuron_lib_, "NeuronCompilation_setPriority");
    fn_set_preference_ = (FnNeuronCompilation_setPreference)dlsym(
        neuron_lib_, "NeuronCompilation_setPreference");
    fn_set_optimization_ = (FnNeuronCompilation_setOptimizationString)dlsym(
        neuron_lib_, "NeuronCompilation_setOptimizationString");
    fn_compilation_finish_ = (FnNeuronCompilation_finish)dlsym(
        neuron_lib_, "NeuronCompilation_finish");

    // Load Execution functions
    fn_execution_create_ = (FnNeuronExecution_create)dlsym(
        neuron_lib_, "NeuronExecution_create");
    fn_execution_free_ = (FnNeuronExecution_free)dlsym(
        neuron_lib_, "NeuronExecution_free");
    fn_set_input_ = (FnNeuronExecution_setInput)dlsym(
        neuron_lib_, "NeuronExecution_setInput");
    fn_set_output_ = (FnNeuronExecution_setOutput)dlsym(
        neuron_lib_, "NeuronExecution_setOutput");
    fn_compute_ = (FnNeuronExecution_compute)dlsym(
        neuron_lib_, "NeuronExecution_compute");

    // Verify critical functions
    if (!fn_model_create_ || !fn_model_finish_ ||
        !fn_compilation_create_ || !fn_compilation_finish_ ||
        !fn_execution_create_ || !fn_compute_) {
        LOG_ERROR("Failed to load critical Neuron API functions");
        return false;
    }

    LOG_INFO("Neuron API loaded successfully");
    return true;
}

bool NeuronExecutor::LoadDLAFile() {
    LOG_INFO("Loading DLA file: " << model_path_);

    dla_fd_ = open(model_path_.c_str(), O_RDONLY);
    if (dla_fd_ < 0) {
        LOG_ERROR("Failed to open DLA file: " << model_path_);
        return false;
    }

    struct stat sb;
    if (fstat(dla_fd_, &sb) < 0) {
        LOG_ERROR("Failed to get DLA file size");
        return false;
    }

    dla_size_ = sb.st_size;
    LOG_INFO("DLA file size: " << (dla_size_ / 1024.0 / 1024.0) << " MB");

    dla_buffer_ = mmap(nullptr, dla_size_, PROT_READ, MAP_SHARED, dla_fd_, 0);
    if (dla_buffer_ == MAP_FAILED) {
        LOG_ERROR("Failed to mmap DLA file");
        dla_buffer_ = nullptr;
        return false;
    }

    LOG_INFO("DLA file mapped successfully");
    return true;
}

bool NeuronExecutor::CreateModel() {
    LOG_INFO("Creating Neuron Model...");

    int err = fn_model_create_(&model_);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("NeuronModel_create failed: " << err);
        return false;
    }

    // Track operand indices
    std::vector<uint32_t> input_indices;
    std::vector<uint32_t> output_indices;

    // Add input operands
    for (size_t i = 0; i < input_shapes_.size(); i++) {
        NeuronOperandType operand_type;
        operand_type.type = input_type_;
        operand_type.scale = 0.0f;
        operand_type.zeroPoint = 0;
        operand_type.dimensionCount = input_shapes_[i].size();
        operand_type.dimensions = input_shapes_[i].data();

        err = fn_add_operand_(model_, &operand_type);
        if (err != NEURON_NO_ERROR) {
            LOG_ERROR("Failed to add input operand " << i);
            return false;
        }

        input_indices.push_back(i);
        LOG_INFO("  Added input " << i << ": shape [" <<
                 JoinString(input_shapes_[i], ",") << "], size=" <<
                 GetInputSize(i) << " bytes");
    }

    // Add extension operand for compiled network
    int32_t extension_operand_type = 0;
    if (fn_get_ext_operand_type_) {
        err = fn_get_ext_operand_type_(model_,
            RESTORE_DLA_EXTENSION_NAME,
            RESTORE_DLA_EXTENSION_OPERAND_TYPE,
            &extension_operand_type);
        if (err != NEURON_NO_ERROR) {
            LOG_WARNING("getExtensionOperandType failed, using default type");
            extension_operand_type = 0;
        }
    }

    NeuronOperandType ext_operand_type;
    ext_operand_type.type = extension_operand_type;
    ext_operand_type.scale = 0.0f;
    ext_operand_type.zeroPoint = 0;
    ext_operand_type.dimensionCount = 0;
    ext_operand_type.dimensions = nullptr;

    err = fn_add_operand_(model_, &ext_operand_type);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("Failed to add extension operand");
        return false;
    }

    uint32_t extension_index = input_indices.size();
    input_indices.push_back(extension_index);

    // Add output operands
    for (size_t i = 0; i < output_shapes_.size(); i++) {
        NeuronOperandType operand_type;
        operand_type.type = output_type_;
        operand_type.scale = 0.0f;
        operand_type.zeroPoint = 0;
        operand_type.dimensionCount = output_shapes_[i].size();
        operand_type.dimensions = output_shapes_[i].data();

        err = fn_add_operand_(model_, &operand_type);
        if (err != NEURON_NO_ERROR) {
            LOG_ERROR("Failed to add output operand " << i);
            return false;
        }

        output_indices.push_back(input_indices.size() + i);
        LOG_INFO("  Added output " << i << ": shape [" <<
                 JoinString(output_shapes_[i], ",") << "], size=" <<
                 GetOutputSize(i) << " bytes");
    }

    // Set DLA data to extension operand
    err = fn_set_operand_value_(model_, extension_index, dla_buffer_, dla_size_);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("Failed to set DLA operand value");
        return false;
    }

    // Get extension operation type
    int32_t operation_type = 0;
    if (fn_get_ext_operation_type_) {
        err = fn_get_ext_operation_type_(model_,
            RESTORE_DLA_EXTENSION_NAME,
            RESTORE_DLA_EXTENSION_OPERATION_TYPE,
            &operation_type);
        if (err != NEURON_NO_ERROR) {
            LOG_WARNING("getExtensionOperationType failed, using default type");
            operation_type = 0;
        }
    }

    // Add operation
    err = fn_add_operation_(model_,
        operation_type,
        input_indices.size(),
        input_indices.data(),
        output_indices.size(),
        output_indices.data());
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("Failed to add operation");
        return false;
    }

    // Identify inputs and outputs (excluding extension operand)
    err = fn_identify_inputs_outputs_(model_,
        input_indices.size() - 1,  // Exclude DLA operand
        input_indices.data(),
        output_indices.size(),
        output_indices.data());
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("Failed to identify inputs and outputs");
        return false;
    }

    // Finish model
    err = fn_model_finish_(model_);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("NeuronModel_finish failed: " << err);
        return false;
    }

    LOG_INFO("Neuron Model created successfully");
    return true;
}

bool NeuronExecutor::CreateCompilation() {
    LOG_INFO("Creating Neuron Compilation...");

    // Options for high-address memory support
    const char* options = "--apusys-config \"{ \\\"high_addr\\\": true }\"";

    int err = fn_compilation_create_(model_, &compilation_, options);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("NeuronCompilation_create failed: " << err);
        return false;
    }

    // Set priority
    if (fn_set_priority_) {
        fn_set_priority_(compilation_, NEURON_PRIORITY_HIGH);
    }

    // Set preference
    if (fn_set_preference_) {
        fn_set_preference_(compilation_, NEURON_PREFER_SUSTAINED_SPEED);
    }

    // Finish compilation
    err = fn_compilation_finish_(compilation_);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("NeuronCompilation_finish failed: " << err);
        return false;
    }

    LOG_INFO("Neuron Compilation created successfully");
    return true;
}

bool NeuronExecutor::CreateExecution() {
    LOG_INFO("Creating Neuron Execution...");

    int err = fn_execution_create_(compilation_, &execution_);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("NeuronExecution_create failed: " << err);
        return false;
    }

    LOG_INFO("Neuron Execution created successfully");
    return true;
}

bool NeuronExecutor::AllocateBuffers() {
    LOG_INFO("Allocating memory buffers...");

    input_buffers_.resize(input_shapes_.size());
    for (size_t i = 0; i < input_shapes_.size(); i++) {
        size_t size = GetInputSize(i);
        input_buffers_[i].resize(size);
        LOG_INFO("  Allocated input buffer " << i << ": " << size << " bytes");
    }

    output_buffers_.resize(output_shapes_.size());
    for (size_t i = 0; i < output_shapes_.size(); i++) {
        size_t size = GetOutputSize(i);
        output_buffers_[i].resize(size);
        LOG_INFO("  Allocated output buffer " << i << ": " << size << " bytes");
    }

    return true;
}

bool NeuronExecutor::Initialize() {
    LOG_INFO("================================================");
    LOG_INFO("Initializing NeuronExecutor: " << name_);
    LOG_INFO("================================================");

    if (!LoadNeuronAPI()) {
        return false;
    }

    if (!LoadDLAFile()) {
        return false;
    }

    if (!CreateModel()) {
        return false;
    }

    if (!CreateCompilation()) {
        return false;
    }

    if (!CreateExecution()) {
        return false;
    }

    if (!AllocateBuffers()) {
        return false;
    }

    initialized_ = true;
    LOG_INFO("================================================");
    LOG_INFO("NeuronExecutor initialized successfully!");
    LOG_INFO("================================================");

    return true;
}

bool NeuronExecutor::Run(
    const std::vector<const void*>& inputs,
    std::vector<void*>& outputs) {

    if (!initialized_) {
        LOG_ERROR("Executor not initialized");
        return false;
    }

    if (inputs.size() != input_shapes_.size()) {
        LOG_ERROR("Input count mismatch: expected " << input_shapes_.size()
                  << ", got " << inputs.size());
        return false;
    }

    if (outputs.size() != output_shapes_.size()) {
        LOG_ERROR("Output count mismatch: expected " << output_shapes_.size()
                  << ", got " << outputs.size());
        return false;
    }

    // Copy inputs to buffers
    for (size_t i = 0; i < inputs.size(); i++) {
        size_t size = GetInputSize(i);
        memcpy(input_buffers_[i].data(), inputs[i], size);
    }

    // Set inputs
    for (size_t i = 0; i < input_shapes_.size(); i++) {
        int err = fn_set_input_(execution_, i, nullptr,
                                input_buffers_[i].data(),
                                input_buffers_[i].size());
        if (err != NEURON_NO_ERROR) {
            LOG_ERROR("Failed to set input " << i);
            return false;
        }
    }

    // Set outputs
    for (size_t i = 0; i < output_shapes_.size(); i++) {
        int err = fn_set_output_(execution_, i, nullptr,
                                 output_buffers_[i].data(),
                                 output_buffers_[i].size());
        if (err != NEURON_NO_ERROR) {
            LOG_ERROR("Failed to set output " << i);
            return false;
        }
    }

    // Run inference
    int err = fn_compute_(execution_);
    if (err != NEURON_NO_ERROR) {
        LOG_ERROR("NeuronExecution_compute failed: " << err);
        return false;
    }

    // Copy outputs
    for (size_t i = 0; i < outputs.size(); i++) {
        size_t size = GetOutputSize(i);
        memcpy(outputs[i], output_buffers_[i].data(), size);
    }

    return true;
}

size_t NeuronExecutor::GetInputSize(size_t index) const {
    if (index >= input_shapes_.size()) {
        return 0;
    }

    size_t size = GetTypeSize(input_type_);
    for (auto dim : input_shapes_[index]) {
        size *= dim;
    }
    return size;
}

size_t NeuronExecutor::GetOutputSize(size_t index) const {
    if (index >= output_shapes_.size()) {
        return 0;
    }

    size_t size = GetTypeSize(output_type_);
    for (auto dim : output_shapes_[index]) {
        size *= dim;
    }
    return size;
}

size_t NeuronExecutor::GetTypeSize(int type) {
    switch (type) {
        case NEURON_TENSOR_FLOAT32:
        case NEURON_TENSOR_INT32:
            return 4;
        case 4:  // NEURON_TENSOR_FLOAT16
            return 2;
        case 1:  // NEURON_TENSOR_FLOAT16
            return 2;
        default:
            return 4;
    }
}
