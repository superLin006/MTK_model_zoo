/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly prohibited.
 *
 * Copyright  (C) 2022  MediaTek Inc. All rights reserved.
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

#pragma once

#include <memory>
#include "common/Macros.h"
#include "common/SharedLib.h"
#include "neuron/api/RuntimeAPI.h"
#include "neuron/api/Types.h"

namespace mtk::neuropilot {

class NeuronRuntimeLibrary final {
public:
    NeuronRuntimeLibrary();

    ~NeuronRuntimeLibrary();

    bool Initialize();

public:
    int Create(const EnvOptions* optionsToDeprecate, void** runtime) {
        if (UNLIKELY(mFnNeuronRuntime_create == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_create(optionsToDeprecate, runtime);
    }

    int CreateWithOptions(const char* options, const EnvOptions* optionsToDeprecate,
                          void** runtime) {
        if (UNLIKELY(mFnNeuronRuntime_create_with_options == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_create_with_options(options, optionsToDeprecate, runtime);
    }

    int LoadNetworkFromFile(void* runtime, const char* pathToDlaFile) {
        if (UNLIKELY(mFnNeuronRuntime_loadNetworkFromFile == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_loadNetworkFromFile(runtime, pathToDlaFile);
    }

    void Release(void* runtime) {
        if (UNLIKELY(mFnNeuronRuntime_release == nullptr)) {
            return;
        }
        return mFnNeuronRuntime_release(runtime);
    }

    int SetInputShape(void* runtime, uint64_t handle, uint32_t* dims, uint32_t rank) {
        if (UNLIKELY(mFnNeuronRuntime_setInputShape == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_setInputShape(runtime, handle, dims, rank);
    }

    int SetInput(void* runtime, uint64_t handle, const void* buffer, size_t length,
                 BufferAttribute attribute) {
        if (UNLIKELY(mFnNeuronRuntime_setInput == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_setInput(runtime, handle, buffer, length, attribute);
    }

    int SetOutput(void* runtime, uint64_t handle, void* buffer, size_t length,
                  BufferAttribute attribute) {
        if (UNLIKELY(mFnNeuronRuntime_setOutput == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_setOutput(runtime, handle, buffer, length, attribute);
    }

    int SetQoSOption(void* runtime, const QoSOptions* qosOption) {
        if (UNLIKELY(mFnNeuronRuntime_setQoSOption == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_setQoSOption(runtime, qosOption);
    }

    int Inference(void* runtime) {
        if (UNLIKELY(mFnNeuronRuntime_inference == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_inference(runtime);
    }

    int GetInputRank(void* runtime, uint64_t handle, uint32_t* rank) {
        if (UNLIKELY(mFnNeuronRuntime_getInputRank == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_getInputRank(runtime, handle, rank);
    }

    int GetInputSize(void* runtime, uint64_t handle, size_t* size) {
        if (UNLIKELY(mFnNeuronRuntime_getInputSize == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_getInputSize(runtime, handle, size);
    }

    int GetOutputSize(void* runtime, uint64_t handle, size_t* size) {
        if (UNLIKELY(mFnNeuronRuntime_getOutputSize == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_getOutputSize(runtime, handle, size);
    }

    int GetProfiledQoSData(void* runtime, ProfiledQoSData** profiledQoSData,
                           uint8_t* execBoostValue) {
        if (UNLIKELY(mFnNeuronRuntime_getProfiledQoSData == nullptr)) {
            return -1;
        }
        return mFnNeuronRuntime_getProfiledQoSData(runtime, profiledQoSData, execBoostValue);
    }

private:
    std::unique_ptr<SharedLib> mSharedLib;

private:
#define INIT_FUNC(name) Fn##name mFn##name = nullptr;

    INIT_FUNC(NeuronRuntime_create)
    INIT_FUNC(NeuronRuntime_create_with_options)
    INIT_FUNC(NeuronRuntime_loadNetworkFromFile)
    INIT_FUNC(NeuronRuntime_release)
    INIT_FUNC(NeuronRuntime_setInputShape)
    INIT_FUNC(NeuronRuntime_setInput)
    INIT_FUNC(NeuronRuntime_setOutput)
    INIT_FUNC(NeuronRuntime_setQoSOption)
    INIT_FUNC(NeuronRuntime_inference)
    INIT_FUNC(NeuronRuntime_getInputRank)
    INIT_FUNC(NeuronRuntime_getInputSize)
    INIT_FUNC(NeuronRuntime_getOutputSize)
    INIT_FUNC(NeuronRuntime_getProfiledQoSData)

#undef INIT_FUNC

private:
    DISALLOW_COPY_AND_ASSIGN(NeuronRuntimeLibrary);
};

}  // namespace mtk::neuropilot
