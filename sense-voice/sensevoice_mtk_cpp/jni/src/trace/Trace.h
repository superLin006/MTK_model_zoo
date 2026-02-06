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
#include <string>
#include <type_traits>
#include "common/Macros.h"

#include "common/SharedLib.h"

#define NP_ATRACE_BEGIN mtk::neuropilot::ATracerAndroid::Get().BeginSection
#define NP_ATRACE_END mtk::neuropilot::ATracerAndroid::Get().EndSection
#define NP_ATRACE_NAME(name) mtk::neuropilot::NpScopedTrace ___tracer(name)
#define NP_ATRACE_CALL() NP_ATRACE_NAME(__FUNCTION__);

namespace mtk::neuropilot {

class ATracerAndroid {
public:
    static ATracerAndroid& Get() {
        static ATracerAndroid tracerAndroid;
        return tracerAndroid;
    }

    void BeginSection(const char* name) {
        if (mIsEnabled) {
            mFpAtraceBeginSection(name);
        }
    }

    void EndSection() {
        if (mIsEnabled) {
            mFpAtraceEndSection();
        }
    }

private:
    ATracerAndroid();

private:
    using FpIsEnabled = std::add_pointer<bool()>::type;
    using FpBeginSection = std::add_pointer<void(const char*)>::type;
    using FpEndSection = std::add_pointer<void()>::type;

    std::unique_ptr<SharedLib> mSharedLib;
    FpIsEnabled mFpAtraceIsEnabled = nullptr;
    FpBeginSection mFpAtraceBeginSection = nullptr;
    FpEndSection mFpAtraceEndSection = nullptr;

    bool mIsEnabled = false;

private:
    DISALLOW_COPY_AND_ASSIGN(ATracerAndroid);
};

static bool EnableSystrace() {
    static bool enableSystrace = true;
    return enableSystrace;
}

class NpScopedTrace {
public:
    inline explicit NpScopedTrace(const char* name) {
        if (UNLIKELY(EnableSystrace())) {
            NP_ATRACE_BEGIN(name);
        }
    }

    inline ~NpScopedTrace() {
        if (UNLIKELY(EnableSystrace())) {
            NP_ATRACE_END();
        }
    }

private:
    DISALLOW_IMPLICIT_CONSTRUCTORS(NpScopedTrace);
};

}  // namespace mtk::neuropilot
