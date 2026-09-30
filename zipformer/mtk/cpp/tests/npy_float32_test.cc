#include "npy_float32.h"
#include <cassert>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string root = argv[1];
    std::vector<float> values;
    int rows = -1, cols = -1;
    for (const char* name : {"valid.npy", "valid_v2.npy", "valid_v3.npy"}) {
        assert(zipformer::load_npy_float32((root + "/" + name).c_str(), values, rows, cols));
        assert(rows == 2 && cols == 3);
        assert(values == std::vector<float>({1, 2, 3, 4, 5, 6}));
    }
    for (const char* name : {"f64.npy", "big_endian.npy", "fortran.npy", "zero.npy",
                            "three_dim.npy", "truncated.npy", "extra.npy", "huge_shape.npy",
                            "bad_version.npy", "bad_header.npy", "missing.npy"}) {
        assert(!zipformer::load_npy_float32((root + "/" + name).c_str(), values, rows, cols));
        assert(values.empty() && rows == 0 && cols == 0);
    }
}
