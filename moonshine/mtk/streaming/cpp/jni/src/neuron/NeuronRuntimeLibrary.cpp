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

#include "NeuronRuntimeLibrary.h"

namespace mtk::neuropilot {

NeuronRuntimeLibrary::NeuronRuntimeLibrary() { Initialize(); }

NeuronRuntimeLibrary::~NeuronRuntimeLibrary() {}

bool NeuronRuntimeLibrary::Initialize() {
    static const std::string libraries[] = {
        "libneuron_runtime.8.so",
        "libneuron_runtime.7.so",
        "libneuron_runtime.6.so",
        "libneuron_runtime.so",
    };

    for (const auto& lib : libraries) {
        auto loader = SharedLib::Load(lib);
        if (LIKELY(loader)) {
            LOG(INFO) << "dlopen " << lib;
            mSharedLib = std::move(loader);
            break;
        }
    }

    if (UNLIKELY(mSharedLib == nullptr)) {
        LOG(ERROR) << "Load Neuron runtime shared library failed.";
        return false;
    }

#define LOAD(Name, name) mFn##Name = mSharedLib->LoadFunc<Fn##Name>(#name);

    LOAD(NeuronRuntime_create, NeuronRuntime_create)
    LOAD(NeuronRuntime_create_with_options, NeuronRuntime_create_with_options)
    LOAD(NeuronRuntime_loadNetworkFromFile, NeuronRuntime_loadNetworkFromFile)
    LOAD(NeuronRuntime_release, NeuronRuntime_release)
    LOAD(NeuronRuntime_setInputShape, NeuronRuntime_setInputShape)
    LOAD(NeuronRuntime_setInput, NeuronRuntime_setInput)
    LOAD(NeuronRuntime_setOutput, NeuronRuntime_setOutput)
    LOAD(NeuronRuntime_setQoSOption, NeuronRuntime_setQoSOption)
    LOAD(NeuronRuntime_inference, NeuronRuntime_inference)
    LOAD(NeuronRuntime_getInputRank, NeuronRuntime_getInputRank)
    LOAD(NeuronRuntime_getInputSize, NeuronRuntime_getInputSize)
    LOAD(NeuronRuntime_getOutputSize, NeuronRuntime_getOutputSize)
    LOAD(NeuronRuntime_getProfiledQoSData, NeuronRuntime_getProfiledQoSData)

#undef LOAD
    return true;
}

}  // namespace mtk::neuropilot
