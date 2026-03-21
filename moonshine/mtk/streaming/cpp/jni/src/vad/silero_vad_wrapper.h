/* silero_vad_wrapper.h - Thin wrapper around SileroVad for Moonshine streaming
 *
 * Provides a simple interface: feed raw PCM chunks, get back a boolean
 * indicating whether a silence boundary was detected (should trigger decoder).
 *
 * The underlying SileroVad returns per-chunk speech probability (0 or 1).
 * This wrapper implements a small state machine:
 *   - Track consecutive silence chunks
 *   - When silence exceeds min_silence_ms, fire a "trigger" once per utterance
 *   - After trigger, wait for next speech before arming again
 */

#pragma once

#include "vad/silero-vad.h"

namespace moonshine {

class SileroVadWrapper {
public:
    // Parameters:
    //   sample_rate      - must be 16000
    //   window_ms        - VAD window size in ms (32ms = 512 samples)
    //   threshold        - speech probability threshold (0.5)
    //   min_silence_ms   - consecutive silence ms before triggering (400ms)
    //   speech_pad_ms    - padding around speech segments (30ms)
    //   min_speech_ms    - minimum speech duration before silence can trigger (250ms)
    SileroVadWrapper(int sample_rate = 16000,
                     int window_ms = 32,
                     float threshold = 0.5f,
                     int min_silence_ms = 400,
                     int speech_pad_ms = 30,
                     int min_speech_ms = 250);

    ~SileroVadWrapper() = default;

    // Returns true if VAD detects a silence boundary (end of speech segment).
    // Caller should trigger decoder when this returns true.
    // samples: raw 16kHz PCM float32, num_samples can be any size.
    // Internally processes samples in window_size_samples_ windows.
    bool ShouldTrigger(const float* samples, int num_samples);

    // Reset state for new stream
    void Reset();

    bool IsLoaded() const { return vad_.is_loaded(); }

private:
    SileroVad vad_;

    // Window size in samples (e.g., 32ms * 16 = 512)
    int window_size_samples_;
    int sr_per_ms_;

    // State machine
    bool in_speech_         = false;  // currently in a speech segment
    bool triggered_         = false;  // have we fired trigger for this segment?
    int  silence_samples_   = 0;      // consecutive silence samples seen
    int  speech_samples_    = 0;      // speech samples accumulated in current seg
    int  min_silence_samples_;
    int  min_speech_samples_;

    // Leftover samples buffer (for samples < window size)
    std::vector<float> remainder_;
};

} // namespace moonshine
