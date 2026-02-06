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

#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>

#include "common/Macros.h"
#include "utils/DumpWorker.h"

namespace mtk::neuropilot {

// Types supported by tensor
typedef enum {
    kNoType = 0,
    kFloat32 = 1,
    kInt32 = 2,
    kUInt8 = 3,
    kInt64 = 4,
    kString = 5,
    kBool = 6,
    kInt16 = 7,
    kComplex64 = 8,
    kInt8 = 9,
    kFloat16 = 10,
    kFloat64 = 11,
    kComplex128 = 12,
    kUInt64 = 13,
    kResource = 14,
    kVariant = 15,
    kUInt32 = 16,
    kUInt16 = 17,
    kInt4 = 18,
} ExecutorDataType;

struct TensorBuffer {
    void* data;
    size_t bytes;  // Size in bytes
    ExecutorDataType type;
};

typedef enum {
    kExecutorSizeError = SIZE_MAX,
} ExecutorStatusCode;

class Executor {
public:
    Executor(const std::string& name) : kName(name) {}

    virtual ~Executor() {}

    virtual bool Load(const std::string& modelPath) = 0;

    virtual bool Initialized() { return mInitiated; }

    virtual bool RunForMultipleInputsOutputs(const std::vector<TensorBuffer>& inputs,
                                             const std::vector<TensorBuffer>& outputs) = 0;

    // Get input tensor byte size by the given index
    virtual size_t GetInputTensorSize(size_t index) = 0;

    // Get output tensor byte size by the given index
    virtual size_t GetOutputTensorSize(size_t index) = 0;

    virtual bool SetInput(size_t index, TensorBuffer buffer) = 0;

    virtual bool GetOutput(size_t index, TensorBuffer buffer) = 0;

    virtual void SetAllowFp16PrecisionForFp32(bool allow) = 0;

    virtual void SetNumThreads(uint32_t num) = 0;

    virtual void Dump(const std::vector<TensorBuffer>& inputs,
                      const std::vector<TensorBuffer>& outputs, size_t number) {
        if (!DumpWorkerInstance.isEnabled()) {
            return;
        }
        for (uint32_t i = 0; i < inputs.size(); i++) {
            DumpWorkerInstance.Dump(kName + "_" + std::to_string(number), inputs[i].data,
                                    inputs[i].bytes, DumpWorker::DumpType::INPUT, i);
        }
        for (uint32_t i = 0; i < outputs.size(); i++) {
            DumpWorkerInstance.Dump(kName + "_" + std::to_string(number), outputs[i].data,
                                    outputs[i].bytes, DumpWorker::DumpType::OUTPUT, i);
        }
    };


protected:
    bool mInitiated = false;

    const std::string kName;

private:
    DISALLOW_COPY_AND_ASSIGN(Executor);
};

}  // namespace mtk::neuropilot
