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

#include "NeuronExecutor.h"

#include <cstring>
#include <iostream>
#include <string>

#include "common/Log.h"

namespace mtk::neuropilot {

NeuronExecutor::~NeuronExecutor() {
    if (mRuntime != nullptr) {
        mNeuronRuntimeLib->Release(mRuntime);
        mRuntime = nullptr;
    }
}

bool NeuronExecutor::Load(const std::string& modelPath) { return true; }

bool NeuronExecutor::RunForMultipleInputsOutputs(const std::vector<TensorBuffer>& inputs,
                                                 const std::vector<TensorBuffer>& outputs) {
    LOG(INFO) << "NeuronExecutor RunForMultipleInputsOutputs";
    // Set input
    for (size_t i = 0; i < inputs.size(); i++) {
        SetInput(i, inputs[i]);
    }

    // Inference
    if (mNeuronRuntimeLib->Inference(mRuntime) != NEURONRUNTIME_NO_ERROR) {
        LOG(ERROR) << "NeuronExecutor fail to inference";
        return false;
    }
    // Get output
    for (size_t i = 0; i < outputs.size(); i++) {
        GetOutput(i, outputs[i]);
    }
    return true;
}

bool NeuronExecutor::Initialize() {
    LOG(INFO) << "NeuronExecutor initialize from model " << kModelPath;

    mNeuronRuntimeLib = std::make_shared<NeuronRuntimeLibrary>();

    EnvOptions envOptions;
    envOptions.deviceKind = kEnvOptHardware;
    envOptions.MDLACoreOption = Dual;
    envOptions.CPUThreadNum = 1;
    envOptions.suppressInputConversion = false;
    envOptions.suppressOutputConversion = false;

    if (mNeuronRuntimeLib->CreateWithOptions(kOptions.length() != 0 ? kOptions.c_str() : "",
                                             &envOptions, &mRuntime) != NEURONRUNTIME_NO_ERROR) {
        LOG(ERROR) << "Failed to create Neuron runtime.";
        return false;
    }

    if (mNeuronRuntimeLib->LoadNetworkFromFile(mRuntime, kModelPath.c_str()) !=
        NEURONRUNTIME_NO_ERROR) {
        LOG(ERROR) << "Failed to load network from file: " << kModelPath;
        mNeuronRuntimeLib->Release(mRuntime);
        mRuntime = nullptr;
        return false;
    }

    LOG(INFO) << "NeuronExecutor get QoS data: " << mGetQoSData;

    mQosOptions.powerPolicy = NEURONRUNTIME_POWER_POLICY_SUSTAINABLE;
    mQosOptions.preference = NEURONRUNTIME_PREFER_PERFORMANCE;
    mQosOptions.boostValue = 100;
    mQosOptions.priority = NEURONRUNTIME_PRIORITY_HIGH;
    if (mNeuronRuntimeLib->SetQoSOption(mRuntime, &mQosOptions) != NEURONRUNTIME_NO_ERROR) {
        LOG(INFO) << "Set QoS Failed";
    };


    size_t i = 0;
    std::string identifier = std::to_string(__COUNTER__) + "_input_";
    while (true) {
        auto size = GetInputTensorSize(i);
        if (size == kExecutorSizeError) {
            break;
        }

        mInputMemory.emplace_back(Memory::Kind::DMABUF, size, identifier + std::to_string(i));
        BufferAttribute bufAttr{
            .ionFd = mInputMemory[i].GetDmaBufFd(),
        };
        mNeuronRuntimeLib->SetInput(mRuntime, i, mInputMemory[i].GetAddr(),
                                    mInputMemory[i].GetSize(), bufAttr);

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

        mOutputMemory.emplace_back(Memory::Kind::DMABUF, size, identifier + std::to_string(i));
        BufferAttribute bufAttr{
            .ionFd = mOutputMemory[i].GetDmaBufFd(),
        };
        mNeuronRuntimeLib->SetOutput(mRuntime, i, mOutputMemory[i].GetAddr(),
                                     mOutputMemory[i].GetSize(), bufAttr);

        LOG(INFO) << "Output " << i << " size: " << size;
        i++;
    }

    return true;
}

size_t NeuronExecutor::GetInputTensorSize(size_t index) {
    size_t size = 0;
    if (mNeuronRuntimeLib->GetInputSize(mRuntime, index, &size) != NEURONRUNTIME_NO_ERROR) {
        return kExecutorSizeError;
    }
    return size;
}

size_t NeuronExecutor::GetOutputTensorSize(size_t index) {
    size_t size = 0;
    if (mNeuronRuntimeLib->GetOutputSize(mRuntime, index, &size) != NEURONRUNTIME_NO_ERROR) {
        return kExecutorSizeError;
    }
    return size;
}

bool NeuronExecutor::SetInput(size_t index, TensorBuffer buffer) {
    if (index >= mInputMemory.size()) {
        LOG(WARNING) << "Invalid input tensor index: " << index;
        return false;
    }
    memcpy(mInputMemory[index].GetAddr(), buffer.data, buffer.bytes);
    return true;
}

bool NeuronExecutor::GetOutput(size_t index, TensorBuffer buffer) {
    if (index >= mOutputMemory.size()) {
        LOG(WARNING) << "Invalid output tensor index:" << index;
        return false;
    }
    memcpy(buffer.data, mOutputMemory[index].GetAddr(), buffer.bytes);
    return true;
}

void NeuronExecutor::SetAllowFp16PrecisionForFp32(bool allow) {
    UNUSED(allow);
    LOG(WARNING) << "NeuronExecutor does not support settingFp16 precision dynamically";
}

}  // namespace mtk::neuropilot
