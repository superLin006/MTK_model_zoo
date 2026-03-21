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
#include "MemAllocator.h"

#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <unistd.h>

namespace mtk::neuropilot {

Memory::~Memory() {
    if (mInfo.NeuronMemory) {
        NeuronMemory_free(mInfo.NeuronMemory);
    }
    if (mInfo.Vaddr && mKind == Kind::DMABUF) {
        munmap(mInfo.Vaddr, mSize);
    }
    if (mInfo.AHardwareBuffer) {
        AHardwareBuffer_unlock(mInfo.AHardwareBuffer, nullptr);
        AHardwareBuffer_release(mInfo.AHardwareBuffer);
    }
    if (mInfo.Fd > 0) {
        close(mInfo.Fd);
    }

}

NeuronMemory* Memory::GetNeuronMemory() const {
    if (mKind != Kind::NEURON_MEMORY) {
        return nullptr;
    }
    return mInfo.NeuronMemory;
}

int Memory::GetDmaBufFd() const {
    return mInfo.Fd;
}

size_t Memory::GetSize() const {
    return mSize;
}

void *Memory::GetAddr() const {
    return mInfo.Vaddr;
}

int Memory::CreateNeuronMemory(size_t size, const std::string& identififer) {
    auto usage = AHARDWAREBUFFER_USAGE_CPU_READ_RARELY | AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY;

    AHardwareBuffer_Desc Desc{
        .width = static_cast<uint32_t>(size),
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = usage,
        .stride = static_cast<uint32_t>(size),
    };
    AHardwareBuffer* ahwBuffer = nullptr;
    if (AHardwareBuffer_allocate(&Desc, &ahwBuffer) != 0) {
        LOG(ERROR) << "Allocate AHardwareBuffer fail";
        return -1;
    }

    NeuronMemory* memory = nullptr;
    NeuronMemory_createFromAHardwareBuffer(ahwBuffer, &memory);


    void* dataBuffer = nullptr;
    AHardwareBuffer_lock(ahwBuffer, usage, -1, NULL, &dataBuffer);

    mInfo.Vaddr = dataBuffer;
    mInfo.AHardwareBuffer = ahwBuffer;
    mInfo.NeuronMemory = memory;
    return 0;
}

int Memory::CreateDmaBuf(size_t size, const std::string& identifier) {
    int fd = open(kDmaDevice.c_str(), O_RDWR);
    if (fd < 0) {
        LOG(ERROR) << "Failed to open " << kDmaDevice;
        return -1;
    }

    struct dma_heap_allocation_data heap_info = {
        .len = size,
        .fd_flags = O_RDWR | O_CLOEXEC,
    };

    if (ioctl(fd, DMA_HEAP_IOCTL_ALLOC, &heap_info) < 0) {
        close(fd);
        LOG(ERROR) << "Failed to allocate DMA heap memory";
        return -1;
    };

    mInfo.Fd = heap_info.fd;

    mInfo.Vaddr = mmap(NULL, mSize, PROT_READ | PROT_WRITE, MAP_SHARED, mInfo.Fd, 0);
    if (mInfo.Vaddr == MAP_FAILED) {
        LOG(ERROR) << "Failed to mmap from fd " << mInfo.Fd;
        mInfo.Vaddr = nullptr;
        return -1;
    }
    return 0;
}

} // mtk::neuropilot