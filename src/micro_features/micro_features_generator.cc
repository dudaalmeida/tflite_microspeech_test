#include "micro_features_generator.h"

#include <Arduino.h>
#include <cmath>
#include <cstring>

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "micro_model_settings.h"

// Configure FFT to output 16 bit fixed point.
#define FIXED_POINT 16
#define NUM_CHROMA_BINS 12

namespace {

FrontendState g_micro_features_state;
bool g_is_first_time = true;

// Function to calculate Zero Crossing Rate (ZCR)
int CalculateZeroCrossingRate(const int16_t* signal, int length) {
  int zero_crossings = 0;
  for (int i = 1; i < length; i++) {
    if ((signal[i - 1] >= 0 && signal[i] < 0) || (signal[i - 1] < 0 && signal[i] >= 0)) {
      zero_crossings++;
    }
  }
  return zero_crossings;
}

// Function to calculate Chroma STFT (simplified version)
void CalculateChroma(const float* fft_values, int fft_size, float* chroma_bins) {
  memset(chroma_bins, 0, sizeof(float) * NUM_CHROMA_BINS);
  for (int i = 0; i < fft_size; i++) {
    float frequency = i * kAudioSampleFrequency / fft_size;
    int chroma_index = static_cast<int>(round(log2(frequency / 440.0) * 12.0)) % NUM_CHROMA_BINS;
    if (chroma_index < 0) chroma_index += NUM_CHROMA_BINS;
    chroma_bins[chroma_index] += fft_values[i];
  }
}

}  // namespace

TfLiteStatus InitializeMicroFeatures(tflite::ErrorReporter* error_reporter) {
  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state, kAudioSampleFrequency)) {
    TF_LITE_REPORT_ERROR(error_reporter, "FrontendPopulateState() failed");
    return kTfLiteError;
  }
  g_is_first_time = true;
  return kTfLiteOk;
}

TfLiteStatus GenerateMicroFeatures(tflite::ErrorReporter* error_reporter,
                                   const int16_t* input, int input_size,
                                   int output_size, uint8_t* output,
                                   size_t* num_samples_read) {
  const int16_t* frontend_input;
  if (g_is_first_time) {
    frontend_input = input;
    g_is_first_time = false;
  } else {
    frontend_input = input + 160;
  }
  FrontendOutput frontend_output = FrontendProcessSamples(
      &g_micro_features_state, frontend_input, input_size, num_samples_read);

  // Calculate Zero Crossing Rate (ZCR)
  int zcr = CalculateZeroCrossingRate(frontend_input, input_size);

  // Calculate Chroma STFT
  float fft_output[frontend_output.size];
  for (int i = 0; i < frontend_output.size; i++) {
    fft_output[i] = frontend_output.values[i] / 32768.0;  // Normalize values
  }
  float chroma_bins[NUM_CHROMA_BINS];
  CalculateChroma(fft_output, frontend_output.size, chroma_bins);

  // Combine features into output buffer
  for (int i = 0; i < frontend_output.size; ++i) {
    constexpr int32_t value_scale = (10 * 255);
    constexpr int32_t value_div = (256 * 26);
    int32_t value = ((frontend_output.values[i] * value_scale) + (value_div / 2)) / value_div;
    if (value < 0) {
      value = 0;
    }
    if (value > 255) {
      value = 255;
    }
    output[i] = value;
  }

  // Append ZCR to the output
  output[frontend_output.size] = zcr;

  // Append Chroma bins to the output
  for (int i = 0; i < NUM_CHROMA_BINS; ++i) {
    output[frontend_output.size + 1 + i] = static_cast<uint8_t>(chroma_bins[i] * 255);
  }

  return kTfLiteOk;
}