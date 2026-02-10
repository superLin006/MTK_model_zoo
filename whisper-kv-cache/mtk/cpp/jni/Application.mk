# Whisper MTK NPU - Application Configuration

# Target architecture
APP_ABI := arm64-v8a

# Platform level (Android 9.0)
APP_PLATFORM := android-28

# STL
APP_STL := c++_static

# CPP flags
APP_CPPFLAGS := -std=c++14 -fexceptions -frtti

# Build parallel jobs
APP_PIE := true

# Short commands
APP_SHORT_COMMANDS := false
