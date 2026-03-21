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

#pragma once

#include <memory>
#include "Executor.h"
#include "neuron/NeuronRuntimeLibrary.h"

#include "utils/MemAllocator.h"

namespace mtk::neuropilot {

class NeuronExecutor : public Executor {
public:
    explicit NeuronExecutor(const std::string& name, const std::string& modelPath)
        : Executor(name), kModelPath(modelPath) {
        mInitiated = Initialize();
    }

    explicit NeuronExecutor(const std::string& name, const std::string& modelPath,
                            const std::string& options)
        : Executor(name), kModelPath(modelPath), kOptions(options) {
        mInitiated = Initialize();
    }

    virtual ~NeuronExecutor();

    virtual bool Load(const std::string& modelPath) override;

    virtual bool RunForMultipleInputsOutputs(const std::vector<TensorBuffer>& inputs,
                                             const std::vector<TensorBuffer>& outputs) override;

    virtual size_t GetInputTensorSize(size_t index) override;

    virtual size_t GetOutputTensorSize(size_t index) override;

    virtual bool SetInput(size_t index, TensorBuffer buffer) override;

    virtual bool GetOutput(size_t index, TensorBuffer buffer) override;

    virtual void SetAllowFp16PrecisionForFp32(bool allow) override;

    virtual void SetNumThreads(uint32_t num) override { UNUSED(num); }

private:
    bool Initialize();

private:
    const std::string kModelPath;

    const std::string kOptions = "--apusys-config \"{ \\\"high_addr\\\": true }\"";

    std::shared_ptr<NeuronRuntimeLibrary> mNeuronRuntimeLib;

    void* mRuntime = nullptr;

    std::vector<Memory> mInputMemory;

    std::vector<Memory> mOutputMemory;

    bool mGetQoSData = true;

    QoSOptions mQosOptions = {};

private:
    DISALLOW_COPY_AND_ASSIGN(NeuronExecutor);
};

}  // namespace mtk::neuropilot
