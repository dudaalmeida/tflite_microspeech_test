/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "recognize_commands.h"
#include <Arduino.h>

#include <limits>

RecognizeCommands::RecognizeCommands(tflite::ErrorReporter* error_reporter,
                                     int32_t average_window_duration_ms,
                                     uint8_t detection_threshold,
                                     int32_t suppression_ms,
                                     int32_t minimum_count)
    : error_reporter_(error_reporter),
      average_window_duration_ms_(average_window_duration_ms),
      detection_threshold_(detection_threshold),
      suppression_ms_(suppression_ms),
      minimum_count_(minimum_count),
      previous_results_(error_reporter) {
  previous_top_label_ = "silence";
  previous_top_label_time_ = std::numeric_limits<int32_t>::min();
}

TfLiteStatus RecognizeCommands::ProcessLatestResults(
    const TfLiteTensor* latest_results, const int32_t current_time_ms,
    const char** found_command, uint8_t* score, bool* is_new_command) {
    log_d("Chegou até aqui no command recognizer");
  if ((latest_results->dims->size != 2) ||
      (latest_results->dims->data[0] != 1) ||
      (latest_results->dims->data[1] != kCategoryCount)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "The results for recognition should contain %d elements, but there are "
        "%d in an %d-dimensional shape",
        kCategoryCount, latest_results->dims->data[1],
        latest_results->dims->size);
    return kTfLiteError;
  }
  //log_d("Chegou até aqui no command recognizer");

  if (latest_results->type != kTfLiteUInt8) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "The results for recognition should be uint8 elements, but are %d",
        latest_results->type);
    return kTfLiteError;
  }

  //log_d("Chegou até aqui no command recognizer");

  if ((!previous_results_.empty()) &&
      (current_time_ms < previous_results_.front().time_)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Results must be fed in increasing time order, but received a "
        "timestamp of %d that was earlier than the previous one of %d",
        current_time_ms, previous_results_.front().time_);
    return kTfLiteError;
  }

  //log_d("Chegou até aqui no command recognizer");

  // Add the latest results to the head of the queue.
  previous_results_.push_back({current_time_ms, latest_results->data.uint8});

  log_d("Adicionando resultado: Tempo: %i", current_time_ms);


  // Prune any earlier results that are too old for the averaging window.
  //logs adicionados
  const int64_t time_limit = current_time_ms - average_window_duration_ms_;
  log_d("Iniciando limpeza do buffer. Tamanho atual: %i", previous_results_.size());
  while ((!previous_results_.empty()) &&
  //while ((previous_results_.size() > 4) &&
        previous_results_.front().time_ < time_limit) {
    log_d("Removendo resultado antigo: Tempo: %i", previous_results_.front().time_);
    previous_results_.pop_front();
  }
  log_d("Fim da limpeza do buffer. Tamanho atual: %i", previous_results_.size());


  // If there are too few results, assume the result will be unreliable and
  // bail.
  const int64_t how_many_results = previous_results_.size();
  const int64_t earliest_time = previous_results_.front().time_;
  const int64_t samples_duration = current_time_ms - earliest_time;

  //O código está ficando preso aqui
  log_d("how many results: %i", how_many_results);
  log_d("minimum_count_: %i", minimum_count_);
  log_d("samples_duration : %i", samples_duration);
  log_d("average_window_duration_ms_: %i", average_window_duration_ms_);

  log_d("Conteúdo de previous_results_: ");
  for (int offset = 0; offset < previous_results_.size(); ++offset) {
    log_d("Tempo: %i", previous_results_.from_front(offset).time_);
  }

  //log_d("\n\n\n\n\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n\n\n\n");

  //if ((how_many_results < minimum_count_) ||
  if ((how_many_results < 0) &&
      (samples_duration < (average_window_duration_ms_ / 4))) {
    *found_command = previous_top_label_;
    *score = 0;
    *is_new_command = false;
    return kTfLiteOk;
  }

  //log_d("\n\n\n\n\n yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy \n\n\n\n\n");

  // Calculate the average score across all the results in the window.
  int32_t average_scores[kCategoryCount];
  for (int offset = 0; offset < previous_results_.size(); ++offset) {
    PreviousResultsQueue::Result previous_result =
        previous_results_.from_front(offset);
    const uint8_t* scores = previous_result.scores_;
    for (int i = 0; i < kCategoryCount; ++i) {
      if (offset == 0) {
        average_scores[i] = scores[i];
      } else {
        average_scores[i] += scores[i];
      }
    }
  }
  for (int i = 0; i < kCategoryCount; ++i) {
    average_scores[i] /= how_many_results;
  }

  // Logando a pontuação média de cada categoria
  log_d("Pontuação média por categoria:");
  int32_t a=0;
  for (int i = 0; i < kCategoryCount; ++i) {
    log_d("Categoria ");
    log_d("%i",i);
    log_d(": ");
    a = average_scores[i];
    log_d("%i", a);
  }

  //log_d("\n\n\n\n\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n\n\n\n");

  // Find the current highest scoring category.
  int current_top_index = 0;
  int32_t current_top_score = 0;
  for (int i = 0; i < kCategoryCount; ++i) {
    if (average_scores[i] > current_top_score) {
      current_top_score = average_scores[i];
      current_top_index = i;
    }
  }
  const char* current_top_label = kCategoryLabels[current_top_index];

  // If we've recently had another label trigger, assume one that occurs too
  // soon afterwards is a bad result.
  int64_t time_since_last_top;
  if ((previous_top_label_ == kCategoryLabels[0]) ||
      (previous_top_label_time_ == std::numeric_limits<int32_t>::min())) {
    time_since_last_top = std::numeric_limits<int32_t>::max();
  } else {
    time_since_last_top = current_time_ms - previous_top_label_time_;
  }
  if ((current_top_score > detection_threshold_) &&
      ((current_top_label != previous_top_label_) ||
       (time_since_last_top > suppression_ms_))) {
    previous_top_label_ = current_top_label;
    previous_top_label_time_ = current_time_ms;
    *is_new_command = true;
  } else {
    *is_new_command = false;
  }
  *found_command = current_top_label;
  *score = current_top_score;

  log_d("Comand recognized");

  return kTfLiteOk;
}
