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

#include "utils/Utils.h"
#include "easyloggingpp/easylogging++.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace mtk::neuropilot {

std::vector<float> ReadFileIntoFloatVector(const std::string& filePath) {
    std::vector<float> output;

    std::ifstream ifs(filePath, std::ifstream::binary);
    if (!ifs) {
        LOG(ERROR) << "Diffusion::Fail to read " << filePath;
        return {};
    }
    LOG(INFO) << "Diffusion::Read " << filePath << " successfully";
    ifs.seekg(0, ifs.end);
    size_t length = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    std::vector<char> fileBuffer;
    fileBuffer.reserve(length);
    // read the data:
    fileBuffer.insert(fileBuffer.begin(), std::istreambuf_iterator<char>(ifs),
                      std::istreambuf_iterator<char>());

    ifs.close();
    float* vectorData = reinterpret_cast<float*>(fileBuffer.data());
    for (int i = 0; i < length / sizeof(float); i++) {
        output.push_back(vectorData[i]);
    }
    return output;
}

std::vector<int32_t> ReadFileIntoIntVector(const std::string& filePath) {
    std::vector<int32_t> output;

    std::ifstream ifs(filePath, std::ifstream::binary);
    if (!ifs) {
        LOG(ERROR) << "Diffusion::Fail to read " << filePath;
        return {};
    }
    LOG(INFO) << "Diffusion::Read " << filePath << " successfully";
    ifs.seekg(0, ifs.end);
    size_t length = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    std::vector<char> fileBuffer;
    fileBuffer.reserve(length);
    // read the data:
    fileBuffer.insert(fileBuffer.begin(), std::istreambuf_iterator<char>(ifs),
                      std::istreambuf_iterator<char>());

    ifs.close();
    int32_t* vectorData = reinterpret_cast<int32_t*>(fileBuffer.data());
    for (int i = 0; i < length / sizeof(int32_t); i++) {
        output.push_back(vectorData[i]);
    }
    return output;
}

std::vector<uint8_t> ReadFileIntoVector(const std::string& filePath) {
    std::vector<uint8_t> output;

    std::ifstream ifs(filePath, std::ifstream::binary);
    if (!ifs) {
        LOG(ERROR) << "Diffusion::Fail to read " << filePath;
        return {};
    }
    LOG(INFO) << "Diffusion::Read " << filePath << " successfully";
    ifs.seekg(0, ifs.end);
    size_t length = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    output.reserve(length);
    // read the data:
    output.insert(output.begin(), std::istream_iterator<uint8_t>(ifs),
                  std::istream_iterator<uint8_t>());

    return output;
}

template<typename T>
void saveToBinaryFile(const std::vector <T> input, const std::string &filename) {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
        return;
    }
    // Write vector data to file
    ofs.write(reinterpret_cast<const char *>(input.data()), input.size() * sizeof(T));

    // Close the file
    ofs.close();
}

template
void saveToBinaryFile(const std::vector<float> input, const std::string &filename);

template
void saveToBinaryFile(const std::vector<int> input, const std::string &filename);


namespace debug2 {
    std::vector<float> ReadFileIntoFloatVector(const std::string &filePath) {
        std::ifstream ifs(filePath, std::ifstream::binary);
        if (!ifs) {
            LOG(ERROR) << "Diffusion::Fail to read " << filePath;
            return {};
        }
        LOG(INFO) << "Diffusion::Read " << filePath << " successfully";

        ifs.seekg(0, std::ios::end);
        std::vector<float>::size_type length = ifs.tellg() / sizeof(float);
        ifs.seekg(0, std::ios::beg);

        std::vector<float> output(length);
        ifs.read(reinterpret_cast<char*>(output.data()), length * sizeof(float));

        if (!ifs) {
            LOG(ERROR) << "Diffusion::Fail to read " << filePath;
            return {};
        }
        return output;
    }

    std::vector<int> ReadFileIntoIntVector(const std::string &filePath) {
        std::ifstream ifs(filePath, std::ifstream::binary);
        if (!ifs) {
            LOG(ERROR) << "Diffusion::Fail to read " << filePath;
            return {};
        }
        LOG(INFO) << "Diffusion::Read " << filePath << " successfully";

        ifs.seekg(0, std::ios::end);
        std::vector<int>::size_type length = ifs.tellg() / sizeof(int);
        ifs.seekg(0, std::ios::beg);

        std::vector<int> output(length);
        ifs.read(reinterpret_cast<char*>(output.data()), length * sizeof(int));

        if (!ifs) {
            LOG(ERROR) << "Diffusion::Fail to read " << filePath;
            return {};
        }
        return output;
    }

    template<typename T>
    void saveToBinaryFile(const std::vector <T> input, const std::string &filename) {
        std::ofstream ofs(filename, std::ios::out | std::ios::binary);
        if (!ofs.is_open()) {
            return;
        }
        // Write vector data to file
        ofs.write(reinterpret_cast<const char *>(input.data()), input.size() * sizeof(T));

        // Close the file
        ofs.close();
    }

    template
    void saveToBinaryFile(const std::vector<float> input, const std::string &filename);

    template
    void saveToBinaryFile(const std::vector<int> input, const std::string &filename);
}

}  // namespace mtk::neuropilot
