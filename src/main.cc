/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <Arduino.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>

#include <esp_task_wdt.h>

#include "C:/Users/eduarda.almeida/Desktop/esp32-tensorflow-microspeech/lib/tfmicro/third_party/kissfft/kiss_fft.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_features/micro_model_settings.h"
#include "micro_features/model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "C:\Users\eduarda.almeida\Desktop\esp32-tensorflow-microspeech\lib\tfmicro\tensorflow\lite\micro\micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

#include "esp_heap_caps.h"

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

#define OLED_RESET 4
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

#define NUMFLAKES     10 // Number of snowflakes in the animation example

#define LOGO_HEIGHT   16
#define LOGO_WIDTH    16
static const unsigned char PROGMEM logo_bmp[] =
{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
};

//SSD1306Wire  display(0x3c, 21, 22);

uint32_t inferenceCounter = 0;
uint32_t inferenceStart = 0;
String inferencePerSecond = "N/A";

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
//constexpr int kTensorArenaSize = 10 * 1024;
constexpr int kTensorArenaSize = 90 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
uint8_t feature_buffer[kFeatureElementCount];
//uint8_t* model_input_buffer = nullptr;
float* model_input_buffer = nullptr; // Se o modelo espera float32

}  // namespace

void intro(const char *message);

// The name of this function is important for Arduino compatibility.
void setup() {
  if (psramFound()) {
   log_d("PSRAM habilitada e detectada!");
  } else {
   log_d("Erro: PSRAM não detectada.");
  }
  //static SSD1306Wire  display(0x3c, 21, 22);
  //display.init();
  //display.clear();
  //display.setFont(ArialMT_Plain_24);
  //display.display();
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.display();
  log_d("Total heap: %d", ESP.getHeapSize());
  log_d("Free heap: %d", ESP.getFreeHeap());
  //log_d("Total PSRAM: %d", ESP.getPsramSize());
  //log_d("Free PSRAM: %d", ESP.getFreePsram());

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::ops::micro::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  #define OPERATIONS_NBR  11
  static tflite::MicroMutableOpResolver<OPERATIONS_NBR> micro_op_resolver;
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddSqueeze();
  micro_op_resolver.AddMaxPool2D();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  //if ((model_input->dims->size != 3) || (model_input->dims->data[0] != NULL) ||
  if ((model_input->dims->size != 3) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    /*
    TF_LITE_REPORT_ERROR(error_reporter,
                     "Tensor dimensions: size=%d, batch=%d, slice_count=%d, slice_size=%d, type=%d",
                     model_input->dims->size,
                     model_input->dims->data[0],
                     model_input->dims->data[1],
                     model_input->dims->data[2],
                     model_input->type);
  */
   /*
    TF_LITE_REPORT_ERROR(error_reporter, "Expected slice_count: %d, slice_size: %d",
                     kFeatureSliceCount, kFeatureSliceSize);

    */
    return;
  }
  //model_input_buffer = model_input->data.uint8;
  model_input_buffer = model_input->data.f;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
  //esp_task_wdt_deinit();

}

// The name of this function is important for Arduino compatibility.
void loop() {  
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    log_d("Feature generation failed");
    return;
  }

  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    //model_input_buffer[i] = feature_buffer[i];
    model_input_buffer[i] = (feature_buffer[i]); 
    log_d("model_input_buffer[i]: %i",  model_input_buffer[i]);
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  log_d("invoke_status: %i", invoke_status);
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    log_d("Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  log_d("Output: %i",output->bytes);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  inferenceCounter++;
  if (inferenceCounter > 10) {
    inferencePerSecond = String((1000.0 * inferenceCounter) / (millis() - inferenceStart)) + "ips";
    inferenceStart = millis();
    inferenceCounter = 0;
  }


  //display.clear();
  //display.setFont(ArialMT_Plain_10);
  //display.drawString(0, 10, inferencePerSecond);
  //display.drawString(0, 20, String(inferenceCounter));
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  //RespondToCommand(error_reporter, &display, current_time, found_command, score,
  //                 is_new_command);
  display.display();
  log_d("Comando: %s \n", found_command);
  log_d("Score: %d \n", score);
  intro(found_command);
}

void intro(const char *message) {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print(F("Message: "));
  display.println(*message);
  
  //display.setCursor(20, 20);
  //display.println(F("Display"));
  //display.setCursor(15, 40);
  //display.println(F("Tutorial"));
  
  display.display();// Show initial text
  delay(1000);
}
