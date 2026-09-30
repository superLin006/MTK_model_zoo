#ifndef ZIPFORMER_NPY_FLOAT32_H_
#define ZIPFORMER_NPY_FLOAT32_H_

#include <cstdint>
#include <fstream>
#include <limits>
#include <regex>
#include <string>
#include <vector>

namespace zipformer {

// Decoder embeddings must be a two-dimensional, little-endian float32 C array.
inline bool load_npy_float32(const char* path, std::vector<float>& data,
                             int& rows, int& cols) {
    data.clear();
    rows = cols = 0;
    if (!path || sizeof(float) != 4) return false;
    const uint16_t native_endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&native_endian) != 1) return false;
    std::ifstream file(path, std::ios::binary);
    uint8_t prefix[8] = {};
    if (!file.read(reinterpret_cast<char*>(prefix), sizeof(prefix)) ||
        prefix[0] != 0x93 || std::string(reinterpret_cast<char*>(prefix + 1), 5) != "NUMPY" ||
        prefix[6] < 1 || prefix[6] > 3 || prefix[7] != 0) return false;

    uint8_t length_bytes[4] = {};
    const unsigned length_size = prefix[6] == 1 ? 2 : 4;
    if (!file.read(reinterpret_cast<char*>(length_bytes), length_size)) return false;
    uint32_t header_length = 0;
    for (unsigned i = 0; i < length_size; ++i) {
        header_length |= static_cast<uint32_t>(length_bytes[i]) << (8 * i);
    }
    if (header_length == 0 || header_length > 1024 * 1024) return false;
    std::string header(header_length, '\0');
    if (!file.read(&header[0], header_length)) return false;
    const std::regex dtype(R"(['"]descr['"]\s*:\s*['"](?:<|=)f4['"])" );
    const std::regex order(R"(['"]fortran_order['"]\s*:\s*False\s*[,}])" );
    const std::regex shape(R"(['"]shape['"]\s*:\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*\))" );
    std::smatch match;
    if (!std::regex_search(header, dtype) || !std::regex_search(header, order) ||
        !std::regex_search(header, match, shape)) return false;

    try {
        const auto r = std::stoull(match[1].str());
        const auto c = std::stoull(match[2].str());
        if (r == 0 || c == 0 || r > std::numeric_limits<int>::max() ||
            c > std::numeric_limits<int>::max() ||
            r > std::numeric_limits<size_t>::max() / c / sizeof(float)) return false;
        const size_t count = static_cast<size_t>(r) * static_cast<size_t>(c);
        const size_t bytes = count * sizeof(float);
        if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;
        const auto data_start = file.tellg();
        file.seekg(0, std::ios::end);
        const auto data_end = file.tellg();
        if (data_start < 0 || data_end < data_start ||
            static_cast<uint64_t>(data_end - data_start) != bytes) return false;
        file.seekg(data_start);
        std::vector<float> loaded(count);
        if (!file.read(reinterpret_cast<char*>(loaded.data()), bytes)) return false;
        data.swap(loaded);
        rows = static_cast<int>(r);
        cols = static_cast<int>(c);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace zipformer
#endif  // ZIPFORMER_NPY_FLOAT32_H_
