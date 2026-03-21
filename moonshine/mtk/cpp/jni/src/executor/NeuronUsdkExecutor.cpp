/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly prohibited.
 *
 * Copyright  (C) 2023  MediaTek Inc. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY ACKNOWLEDGES
 * THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY THIRD PARTY ALL PROPER LICENSES
 * CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK
 * SOFTWARE RELEASES MADE TO RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR
 * STANDARD OR OPEN FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER WILL BE,
 * AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT ISSUE,
 * OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER TO
 * MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek Software")
 * have been modified by MediaTek Inc. All revisions are subject to any receiver's
 * applicable license agreements with MediaTek Inc.
 */

#include "common/Log.h"
#include "utils/MemAllocator.h"
#include "NeuronUsdkExecutor.h"

#include <sys/mman.h> 
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <string>

#define RESTORE_DLA_EXTENSION_OPERAND_TYPE   0x0200 // 0x0100
#define RESTORE_DLA_EXTENSION_OPERATION_TYPE 0x0000
#define RESTORE_DLA_EXTENSION_NAME           "com.mediatek.compiled_network"

namespace mtk::neuropilot {

uint32_t GetNeuronTypeSize(int type) {
    int size = 1;
    switch (type) {
        case NEURON_FLOAT32:
        case NEURON_INT32:
        case NEURON_UINT32:
        case NEURON_TENSOR_FLOAT32:
        case NEURON_TENSOR_INT32:
            size = 4;
            break;
        case NEURON_TENSOR_QUANT16_SYMM:
        case NEURON_TENSOR_FLOAT16:
        case NEURON_FLOAT16:
        case NEURON_TENSOR_QUANT16_ASYMM:
            size = 2;
            break;
        case NEURON_TENSOR_BOOL8:
        case NEURON_TENSOR_QUANT8_SYMM:
        case NEURON_TENSOR_QUANT8_ASYMM_SIGNED:
            size = 1;
            break;
        default:
            LOG(ERROR) << "Get Neuron Type Error";
            size = 0;
    }
    return size;
}

bool GetModelInfo(const std::string& modelPath,
                  std::vector<std::vector<uint32_t>>& input,
                  std::vector<std::vector<uint32_t>>& output,
                  std::vector<uint32_t>& reused_size,
                  int &inputType,
                  int &outputType) {

    if (modelPath.find("text_encoder") != std::string::npos) {
        input = {{1, 77}, {1, 1}};
        output = {{1, 77, 768}, {1, 768}};
        inputType = NEURON_TENSOR_INT32;
        outputType = NEURON_TENSOR_FLOAT32;
    } else if (modelPath.find("sensevoice") != std::string::npos) {
        // SenseVoice model configuration
        // Input 0: Audio features [1, 166, 560] float32 (10s audio after subsampling)
        // Input 1-4: prompt IDs [1] int32 (language_id, event_id, event_type_id, text_norm_id)
        // Output: CTC logits [1, 170, 25055] float32
        input = {{1, 166, 560}, {1}, {1}, {1}, {1}};
        output = {{1, 170, 25055}};
        inputType = NEURON_TENSOR_FLOAT32;
        outputType = NEURON_TENSOR_FLOAT32;
        LOG(INFO) << "Using SenseVoice model configuration";
    } else {
        LOG(ERROR) << "Couldn't find the shape info for model";
        return false;
    }
    return true;
}


NeuronUsdkExecutor::NeuronUsdkExecutor(const std::string& name, const std::string& modelPath, const std::string& kOptions, const std::vector<uint32_t>& reusedSize,
                                       std::vector<std::vector<uint32_t>> inputShape, int inputType,
                                       std::vector<std::vector<uint32_t>> outputShape, int outputType)
        : Executor(name), kModelPath(modelPath), kOptions(kOptions), mInputSize(inputShape), mOutputSize(outputShape), mReusedSize(reusedSize),
          mInputType(inputType), mOutputType(outputType) {
    mInitiated = Initialize();
}

NeuronUsdkExecutor::~NeuronUsdkExecutor() {
    if (mExecution != nullptr) {
        NeuronExecution_free(mExecution);
        mExecution = nullptr;
    }
    if (mCompilation != nullptr) {
        NeuronCompilation_free(mCompilation);
        mCompilation = nullptr;
    }
    if (mModel != nullptr) {
        NeuronModel_free(mModel);
        mModel = nullptr;
    }
}

bool NeuronUsdkExecutor::Load(const std::string& modelPath) { return false; }

bool NeuronUsdkExecutor::RunForMultipleInputsOutputs(const std::vector<TensorBuffer>& inputs,
                                                     const std::vector<TensorBuffer>& outputs) {
    LOG(INFO) << "NeuronUsdkExecutor RunForMultipleInputsOutputs";
    // Set input
    // since 3rd, 4th inputs of unet are fixed, skip SetInput() to save memcpy
    size_t setSize = inputs.size() < 2 ? inputs.size() : 2 ;
    for (size_t i = 0; i < setSize; i++) {
        SetInput(i, inputs[i]);
    }


    // Inference
    if (NeuronExecution_compute(mExecution) != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronUsdkExecutor fail to inference";
        return false;
    }

    // Get output
    for (size_t i = 0; i < outputs.size(); i++) {
        GetOutput(i, outputs[i]);
    }

    return true;
}

bool NeuronUsdkExecutor::Initialize() {
    LOG(INFO) << "NeuronUsdkExecutor initialize from model " << kModelPath;

    if ((mInputSize.size() == 0 || mOutputSize.size() == 0) &&
        (!GetModelInfo(kModelPath, mInputSize, mOutputSize, mReusedSize,
                       mInputType, mOutputType))) {
        LOG(ERROR) << "Get Model Shape Fail";
        return false;
    }

    int fd = open(kModelPath.c_str(), O_RDONLY);
    if (fd == -1) {
        LOG(ERROR) << "Open dla file fail";
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        LOG(ERROR) << "fstat fail";
        return false;
    }

    void* addr = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        LOG(ERROR) << "mmap fail";
        return false;
    }

    if (LoadDla(addr, sb.st_size) == false) {
        LOG(ERROR) << "load dla fail";
        return false;
    };

    if (munmap(addr, sb.st_size) == -1) {
        LOG(ERROR) << "munmap fail";
    }

    if (close(fd) == -1) {
        LOG(ERROR) << "close fail";
    }

    size_t i = 0;
    std::string identifier = std::to_string(__COUNTER__) + "_input_";
    while (true) {
        auto size = GetInputTensorSize(i);
        if (size == kExecutorSizeError) {
            break;
        }
        mInputMemory.emplace_back(Memory::Kind::NEURON_MEMORY, size, identifier + std::to_string(i));
        NeuronExecution_setInputFromMemory(mExecution, i, NULL, mInputMemory[i].GetNeuronMemory(),
                                           0, mInputMemory[i].GetSize());

        LOG(INFO) << "Input " << i << " size: " << size;
        i++;
    }

    i = 0;
    identifier = std::to_string(__COUNTER__) + "_output_";
    while (true) {
        auto size = GetOutputTensorSize(i);
        if (size == kExecutorSizeError) {
            break;
        }
        mOutputMemory.emplace_back(Memory::Kind::NEURON_MEMORY, size, identifier + std::to_string(i));
        NeuronExecution_setOutputFromMemory(mExecution, i, NULL, mOutputMemory[i].GetNeuronMemory(),
                                           0, mOutputMemory[i].GetSize());
        LOG(INFO) << "Output " << i << " size: " << size;
        i++;
    }
    return true;
}

bool NeuronUsdkExecutor::LoadDla(void* buffer, size_t size) {
    int err = NeuronModel_create(&mModel);

    std::vector<uint32_t> inputNode;
    std::vector<uint32_t> outputNode;

    for (int i = 0; i < mInputSize.size(); i++) {
        uint32_t inputNum = mInputSize[i].size();
        const uint32_t *dimsInput = mInputSize[i].data();

        NeuronOperandType tensorInputType;
        tensorInputType.type = mInputType;
        tensorInputType.scale = 0.0f;
        tensorInputType.zeroPoint = 0;
        tensorInputType.dimensionCount = inputNum;
        tensorInputType.dimensions = dimsInput;

        err |= NeuronModel_addOperand(mModel, &tensorInputType);
        inputNode.emplace_back(i);
    }

    int32_t operandType = 0;
    const uint16_t network_operand_restore_data = RESTORE_DLA_EXTENSION_OPERAND_TYPE;
    const char* extensionRestroeCompiledNetwork = RESTORE_DLA_EXTENSION_NAME;
    err |= NeuronModel_getExtensionOperandType(mModel, extensionRestroeCompiledNetwork,
                                               network_operand_restore_data, &operandType);


    NeuronOperandType extenOperandType;
    extenOperandType.type = operandType;
    extenOperandType.scale = 0.0f;
    extenOperandType.zeroPoint = 0;
    extenOperandType.dimensionCount = 0;

    err |= NeuronModel_addOperand(mModel, &extenOperandType);
    inputNode.emplace_back(inputNode.size());

    for (int i = 0; i < mOutputSize.size(); i++) {
        uint32_t outputNum = mOutputSize[i].size();
        const uint32_t *dimsOutput = mOutputSize[i].data();

        NeuronOperandType tensorOutputType;
        tensorOutputType.type = mOutputType;
        tensorOutputType.scale = 0.0f;
        tensorOutputType.zeroPoint = 0;
        tensorOutputType.dimensionCount = outputNum;
        tensorOutputType.dimensions = dimsOutput;

        err |= NeuronModel_addOperand(mModel, &tensorOutputType);
        outputNode.emplace_back(i + inputNode.size());
    }


    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "addOperand fail";
        return false;
    }
    err |= NeuronModel_setOperandValue(mModel, inputNode.back(), buffer, size);

    int32_t operationType = 0;
    const uint16_t network_operation_type_restore = RESTORE_DLA_EXTENSION_OPERATION_TYPE;
    err |= NeuronModel_getExtensionOperationType(mModel, extensionRestroeCompiledNetwork,
                                                 network_operation_type_restore, &operationType);

    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "get ExtensionOperationType fail";
        return false;
    }


    // Add extension operation
    err |= NeuronModel_addOperation(mModel, static_cast<NeuronOperationType>(operationType),
                                    inputNode.size(), inputNode.data(),
                                    outputNode.size(), outputNode.data());

    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "get addOperation fail";
        return false;
    }

    // Identify input and output
    err |= NeuronModel_identifyInputsAndOutputs(mModel,
                                                inputNode.size() - 1, inputNode.data(),
                                                outputNode.size(), outputNode.data());

    err |= NeuronModel_finish(mModel);

    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "get model_finish fail";
        return false;
    }

    if (NeuronCompilation_createWithOptions(mModel, &mCompilation,
                                            kOptions.c_str()) != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronCompilation_create fail";
        return false;
    };

    NeuronCompilation_setPriority(mCompilation, NEURON_PRIORITY_HIGH);
    NeuronCompilation_setPreference(mCompilation, NEURON_PREFER_SUSTAINED_SPEED);

    if (kOptions.length()) {
        NeuronCompilation_setOptimizationString(mCompilation, kOptions.c_str());
    }

    if (NeuronCompilation_finish(mCompilation) != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronCompilation_finish fail";
        return false;
    };
    if (NeuronExecution_create(mCompilation, &mExecution) != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronExecution_create fail";
        return false;
    };

    if (NeuronExecution_setBoostHint(mExecution, 100) != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronExecution_setBoostHint fail";
        return false;
    };
    return true;
}

size_t NeuronUsdkExecutor::GetInputTensorSize(size_t index) {
    auto size = GetNeuronTypeSize(mInputType);
    if (index < mInputSize.size() && size) {
        uint32_t s = 1;
        for (auto i : mInputSize[index]) {
            s *= i;
        }
        return s * size;
    } else {
        return kExecutorSizeError;
    }
}

size_t NeuronUsdkExecutor::GetOutputTensorSize(size_t index) {
    auto size = GetNeuronTypeSize(mInputType);
    if (index < mOutputSize.size() && size) {
        uint32_t s = 1;
        for (auto i : mOutputSize[index]) {
            s *= i;
        }
        return s * size;
    } else {
        return kExecutorSizeError;
    }
}

bool NeuronUsdkExecutor::SetInput(size_t index, TensorBuffer buffer) {
    if (index >= mInputMemory.size()) {
        LOG(WARNING) << "Invalid input tensor index: " << index;
        return false;
    }
    memcpy(mInputMemory[index].GetAddr(), buffer.data, buffer.bytes);
    return true;
}

bool NeuronUsdkExecutor::GetOutput(size_t index, TensorBuffer buffer) {
    if (index >= mOutputMemory.size()) {
        LOG(WARNING) << "Invalid output tensor index:" << index;
        return false;
    }
    memcpy(buffer.data, mOutputMemory[index].GetAddr(), buffer.bytes);
    return true;
}

void NeuronUsdkExecutor::SetAllowFp16PrecisionForFp32(bool allow) {
    UNUSED(allow);
    LOG(WARNING) << "NeuronUsdkExecutor does not support settingFp16 precision dynamically";
}

}  // namespace mtk::neuropilot

