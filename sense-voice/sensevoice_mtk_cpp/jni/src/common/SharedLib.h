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

#include "common/Log.h"
#include "common/Macros.h"
#include <dlfcn.h>
#include <memory>
#include <string>

namespace mtk::neuropilot {
// SharedLib dlopen share library.
class SharedLib {
public:
    static constexpr bool kShouldCloseLib = true;

    // Return nullptr if the dlopen is failed.
    // Return a SharedLib instance of the given library name.
    template <bool kLogError = true>
    static std::unique_ptr<SharedLib> Load(const std::string& name,
                                           bool closeLib = kShouldCloseLib) {
        auto lib = LoadImpl(name, closeLib);
        if constexpr (kLogError) {
            if (UNLIKELY(lib == nullptr)) {
                LOG(ERROR) << "Could not load " << name;
                auto error = dlerror();
                if (UNLIKELY(error != nullptr)) {
                    LOG(ERROR) << error;
                }
            }
        }
        return lib;
    }

    ~SharedLib() {
        DCHECK(mHandle != nullptr);
        dlclose(mHandle);
    }

public:
    // Return a function pointer of the given symbol name.
    // Abort if NullCheck is enabled and the given symbol name is not found in the library.
    template <typename FuncType, bool NullCheck = true>
    FuncType LoadFunc(const char* symbol) const {
        auto fn = LoadFuncImpl(symbol, NullCheck);

        if constexpr (NullCheck) {
            DCHECK(fn != nullptr) << symbol;
        }

        return reinterpret_cast<FuncType>(fn);
    }

    // Return a function pointer of the given symbol name.
    // Return nullptr if the given symbol name is not found in the library.
    template <typename FuncType>
    FuncType LoadWeakFunc(const char* symbol) const {
        return LoadFunc<FuncType, /* NullCheck */ false>(symbol);
    }

private:
    explicit SharedLib(void* handle) : mHandle(handle) { DCHECK(mHandle != nullptr); }

    void* GetHandle() const { return mHandle; }

    static std::unique_ptr<SharedLib> LoadImpl(const std::string& name, bool closeLib) {
        auto flag = RTLD_LAZY | RTLD_LOCAL;
        if (!closeLib) {
            flag = flag | RTLD_NODELETE;
        }
        if (auto handle = dlopen(name.c_str(), flag)) {
            return std::unique_ptr<SharedLib>(new SharedLib(handle));
        }
        return nullptr;
    }

    void* LoadFuncImpl(const char* symbol, bool logError) const {
        DCHECK(symbol != nullptr);

        // Clear any existing error.
        dlerror();

        auto fn = dlsym(GetHandle(), symbol);

        auto error = dlerror();

        if (UNLIKELY(error != nullptr)) {
            if (logError) {
                LOG(ERROR) << error;
            }
            return nullptr;
        }

        return fn;
    }

private:
    void* mHandle = nullptr;

private:
    DISALLOW_IMPLICIT_CONSTRUCTORS(SharedLib);
};

}  // namespace mtk::neuropilot
