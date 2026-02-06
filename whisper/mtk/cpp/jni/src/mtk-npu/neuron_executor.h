/**
 * Neuron Executor - New Neuron API Wrapper
 *
 * Wraps the new Neuron API (NeuronModel, NeuronCompilation, NeuronExecution)
 * for loading and running pre-compiled DLA models on MTK NPU.
 */

#ifndef NEURON_EXECUTOR_H
#define NEURON_EXECUTOR_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Forward declarations for Neuron API types
typedef struct NeuronModel NeuronModel;
typedef struct NeuronCompilation NeuronCompilation;
typedef struct NeuronExecution NeuronExecution;

// Neuron API constants
#define NEURON_NO_ERROR 0
#define NEURON_TENSOR_FLOAT32 2
#define NEURON_TENSOR_INT32 3
#define NEURON_INT32 3

// Priority options
#define NEURON_PRIORITY_LOW 90
#define NEURON_PRIORITY_MEDIUM 100
#define NEURON_PRIORITY_HIGH 110

// Preference options
#define NEURON_PREFER_LOW_POWER 0
#define NEURON_PREFER_FAST_SINGLE_ANSWER 1
#define NEURON_PREFER_SUSTAINED_SPEED 2
#define NEURON_PREFER_TURBO_BOOST 3

// Extension for compiled network
#define RESTORE_DLA_EXTENSION_NAME "com.mediatek.compiled_network"
#define RESTORE_DLA_EXTENSION_OPERAND_TYPE 0x0200
#define RESTORE_DLA_EXTENSION_OPERATION_TYPE 0x0000

// Neuron API function pointers
typedef int (*FnNeuronModel_create)(NeuronModel** model);
typedef void (*FnNeuronModel_free)(NeuronModel* model);
typedef int (*FnNeuronModel_addOperand)(NeuronModel* model, const void* type);
typedef int (*FnNeuronModel_getExtensionOperandType)(NeuronModel* model,
    const char* extensionName, uint16_t operandType, int32_t* outType);
typedef int (*FnNeuronModel_getExtensionOperationType)(NeuronModel* model,
    const char* extensionName, uint16_t operationType, int32_t* outType);
typedef int (*FnNeuronModel_setOperandValue)(NeuronModel* model,
    uint32_t index, const void* buffer, size_t length);
typedef int (*FnNeuronModel_addOperation)(NeuronModel* model,
    int32_t operationType, uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs);
typedef int (*FnNeuronModel_identifyInputsAndOutputs)(NeuronModel* model,
    uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs);
typedef int (*FnNeuronModel_finish)(NeuronModel* model);

typedef int (*FnNeuronCompilation_createWithOptions)(
    NeuronModel* model, NeuronCompilation** compilation, const char* options);
typedef void (*FnNeuronCompilation_free)(NeuronCompilation* compilation);
typedef int (*FnNeuronCompilation_setPriority)(NeuronCompilation* compilation, int priority);
typedef int (*FnNeuronCompilation_setPreference)(NeuronCompilation* compilation, int preference);
typedef int (*FnNeuronCompilation_setOptimizationString)(
    NeuronCompilation* compilation, const char* optimization);
typedef int (*FnNeuronCompilation_finish)(NeuronCompilation* compilation);

typedef int (*FnNeuronExecution_create)(NeuronCompilation* compilation, NeuronExecution** execution);
typedef void (*FnNeuronExecution_free)(NeuronExecution* execution);
typedef int (*FnNeuronExecution_setInput)(NeuronExecution* execution,
    uint32_t index, const void* type, const void* buffer, size_t length);
typedef int (*FnNeuronExecution_setOutput)(NeuronExecution* execution,
    uint32_t index, const void* type, void* buffer, size_t length);
typedef int (*FnNeuronExecution_compute)(NeuronExecution* execution);

/**
 * NeuronExecutor - Loads and executes DLA models using new Neuron API
 */
class NeuronExecutor {
public:
    /**
     * Constructor
     * @param model_path Path to .dla file
     * @param input_shapes Vector of input tensor shapes (e.g., {{1,64,512}, {1,1,64,64}})
     * @param output_shapes Vector of output tensor shapes
     * @param name Executor name (for debugging)
     */
    NeuronExecutor(const std::string& model_path,
                   const std::vector<std::vector<uint32_t>>& input_shapes,
                   const std::vector<std::vector<uint32_t>>& output_shapes,
                   const std::string& name = "NeuronExecutor");

    ~NeuronExecutor();

    // Initialize: load DLA, create model, compilation, and execution
    bool Initialize();

    // Run inference
    bool Run(const std::vector<const void*>& inputs,
             std::vector<void*>& outputs);

    // Get input/output sizes
    size_t GetInputSize(size_t index) const;
    size_t GetOutputSize(size_t index) const;

    // Get number of inputs/outputs
    size_t GetNumInputs() const { return input_shapes_.size(); }
    size_t GetNumOutputs() const { return output_shapes_.size(); }

    // Check if initialized
    bool IsInitialized() const { return initialized_; }

private:
    // Load Neuron API functions from shared library
    bool LoadNeuronAPI();

    // Load DLA file and mmap it
    bool LoadDLAFile();

    // Create Neuron Model with operands
    bool CreateModel();

    // Create Compilation
    bool CreateCompilation();

    // Create Execution
    bool CreateExecution();

    // Allocate memory buffers
    bool AllocateBuffers();

    // Helper: Get size in bytes for a neuron type
    static size_t GetTypeSize(int type);

private:
    std::string model_path_;
    std::string name_;
    std::vector<std::vector<uint32_t>> input_shapes_;
    std::vector<std::vector<uint32_t>> output_shapes_;

    // DLA file mapping
    int dla_fd_ = -1;
    void* dla_buffer_ = nullptr;
    size_t dla_size_ = 0;

    // Neuron API objects
    NeuronModel* model_ = nullptr;
    NeuronCompilation* compilation_ = nullptr;
    NeuronExecution* execution_ = nullptr;

    // Memory buffers
    std::vector<std::vector<uint8_t>> input_buffers_;
    std::vector<std::vector<uint8_t>> output_buffers_;

    // Input/Output operand types
    int input_type_ = NEURON_TENSOR_FLOAT32;
    int output_type_ = NEURON_TENSOR_FLOAT32;

    bool initialized_ = false;

    // Neuron API function pointers
    FnNeuronModel_create fn_model_create_ = nullptr;
    FnNeuronModel_free fn_model_free_ = nullptr;
    FnNeuronModel_addOperand fn_add_operand_ = nullptr;
    FnNeuronModel_getExtensionOperandType fn_get_ext_operand_type_ = nullptr;
    FnNeuronModel_getExtensionOperationType fn_get_ext_operation_type_ = nullptr;
    FnNeuronModel_setOperandValue fn_set_operand_value_ = nullptr;
    FnNeuronModel_addOperation fn_add_operation_ = nullptr;
    FnNeuronModel_identifyInputsAndOutputs fn_identify_inputs_outputs_ = nullptr;
    FnNeuronModel_finish fn_model_finish_ = nullptr;

    FnNeuronCompilation_createWithOptions fn_compilation_create_ = nullptr;
    FnNeuronCompilation_free fn_compilation_free_ = nullptr;
    FnNeuronCompilation_setPriority fn_set_priority_ = nullptr;
    FnNeuronCompilation_setPreference fn_set_preference_ = nullptr;
    FnNeuronCompilation_setOptimizationString fn_set_optimization_ = nullptr;
    FnNeuronCompilation_finish fn_compilation_finish_ = nullptr;

    FnNeuronExecution_create fn_execution_create_ = nullptr;
    FnNeuronExecution_free fn_execution_free_ = nullptr;
    FnNeuronExecution_setInput fn_set_input_ = nullptr;
    FnNeuronExecution_setOutput fn_set_output_ = nullptr;
    FnNeuronExecution_compute fn_compute_ = nullptr;

    void* neuron_lib_ = nullptr;
};

#endif  // NEURON_EXECUTOR_H
