APP_ABI := arm64-v8a
APP_STL := c++_shared
APP_CPPFLAGS := -D__ANDROID__ \
                -D__DEBUG__ \
                -fexceptions \
                -frtti \
                -std=c++17 \
                -Wall \
                -Wno-range-loop-construct

APP_PLATFORM := android-29
