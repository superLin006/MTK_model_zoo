/* silero_vad_wrapper.cpp - Silero VAD wrapper implementation */

#include "vad/silero_vad_wrapper.h"

#include <cstring>
#include <iostream>

namespace moonshine {

SileroVadWrapper::SileroVadWrapper(int sample_rate, int window_ms,
                                   float threshold, int min_silence_ms,
                                   int speech_pad_ms, int min_speech_ms)
    : vad_(sample_rate, window_ms, threshold, min_silence_ms,
           speech_pad_ms, min_speech_ms),
      sr_per_ms_(sample_rate / 1000),
      window_size_samples_(window_ms * (sample_rate / 1000)),
      min_silence_samples_(min_silence_ms * (sample_rate / 1000)),
      min_speech_samples_(min_speech_ms * (sample_rate / 1000))
{
    remainder_.reserve(window_size_samples_);
}

void SileroVadWrapper::Reset() {
    in_speech_       = false;
    triggered_       = false;
    silence_samples_ = 0;
    speech_samples_  = 0;
    remainder_.clear();
}

bool SileroVadWrapper::ShouldTrigger(const float* samples, int num_samples) {
    // Append incoming samples to remainder
    for (int i = 0; i < num_samples; i++) {
        remainder_.push_back(samples[i]);
    }

    bool should_trigger = false;

    // Process complete windows
    while ((int)remainder_.size() >= window_size_samples_) {
        // Build window vector for VAD
        std::vector<float> window(remainder_.begin(),
                                  remainder_.begin() + window_size_samples_);

        // Consume from remainder
        remainder_.erase(remainder_.begin(),
                         remainder_.begin() + window_size_samples_);

        float prob = 0.0f;
        int   flag = 0;   // 0=silence, 1=speech  (SileroVad returns 0 or 1)
        vad_.predict(window, &prob, &flag);

        if (flag == 1) {
            // Speech detected
            in_speech_       = true;
            triggered_       = false;   // arm trigger for next silence
            silence_samples_ = 0;
            speech_samples_ += window_size_samples_;
        } else {
            // Silence
            if (in_speech_) {
                silence_samples_ += window_size_samples_;
                // Fire trigger once when silence exceeds threshold
                // and we have accumulated enough speech
                if (!triggered_ &&
                    silence_samples_ >= min_silence_samples_ &&
                    speech_samples_  >= min_speech_samples_)
                {
                    triggered_      = true;
                    in_speech_      = false;
                    silence_samples_ = 0;
                    speech_samples_  = 0;
                    should_trigger   = true;
                    // (do not break: process remaining windows but
                    //  further triggers in same call will not fire
                    //  since triggered_=true and in_speech_=false)
                }
            }
            // else: not in speech, ignore silence
        }
    }

    return should_trigger;
}

} // namespace moonshine
