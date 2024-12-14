#include "../audio_provider.h"
#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "../micro_features/micro_model_settings.h"

// Definição dos pinos de comunicação I2S
#define I2S_SD 25
#define I2S_WS 26
#define I2S_SCK 33
#define I2S_PORT I2S_NUM_0

// Comprimento do buffer usado para amostrar o áudio
#define bufferLen (kMaxAudioSampleSize)

// Nome do módulo usado para logging
static const char *TAG = "TF_LITE_AUDIO_PROVIDER";

namespace {
  // Buffer para armazenar as amostras de áudio
  int16_t g_audio_output_buffer[kMaxAudioSampleSize];

  // Buffer temporário para armazenar dados enviados
  int16_t g_reference_buffer[kMaxAudioSampleSize];

  // Variável para verificar se a gravação de áudio já foi inicializada
  bool g_is_audio_initialized = false;

  // Timestamp do áudio mais recente (em milissegundos)
  volatile int32_t g_latest_audio_timestamp = 0;
}  // namespace

// Função para inicializar o periférico I2S
static void i2s_init(void) {
  // Configuração do I2S
  const i2s_config_t i2s_config = {
      .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX), // Modo mestre, recepção
      .sample_rate = 16000,                             // Taxa de amostragem em 16 kHz
      .bits_per_sample = i2s_bits_per_sample_t(16),     // 16 bits por amostra
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,      // Canal único (esquerdo)
      .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S), // Formato de comunicação padrão I2S
      .intr_alloc_flags = 0,                            // Sem alocação especial de interrupções
      .dma_buf_count = 4,                               // Número de buffers DMA
      .dma_buf_len = bufferLen,                         // Tamanho do buffer DMA
      .use_apll = false                                 // Sem uso de PLL
  };

  // Configuração dos pinos I2S
  const i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,   // Pino do clock do barramento
      .ws_io_num = I2S_WS,     // Pino de seleção de palavra
      .data_out_num = -1,      // Sem pino de saída de dados
      .data_in_num = I2S_SD    // Pino de entrada de dados
  };

  // Instala o driver I2S com as configurações definidas
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  // Define os pinos do periférico I2S
  i2s_set_pin(I2S_PORT, &pin_config);
  // Zera os buffers DMA
  i2s_zero_dma_buffer(I2S_PORT);
}

// Função para capturar amostras de áudio
static void CaptureSamples(void *arg) {
  size_t bytes_read = 0;
  // Inicializa o periférico I2S
  i2s_init();

  while (1) {
    // Ler diretamente os dados de áudio para o buffer de saída
    i2s_read(I2S_PORT, g_audio_output_buffer, bufferLen * sizeof(int16_t), &bytes_read, portMAX_DELAY);

    if (bytes_read > 0) {
      // Atualiza o timestamp (em ms) com base nos bytes lidos e na frequência de amostragem
      g_latest_audio_timestamp += (1000 * bytes_read / sizeof(int16_t)) / kAudioSampleFrequency;
    } else {
      // Log de erro caso a leitura falhe
      ESP_LOGE(TAG, "Erro na leitura do I2S");
    }

    // Pequeno atraso para evitar sobrecarga da CPU
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  // Deleta a tarefa ao sair do loop
  vTaskDelete(NULL);
}

// Função para inicializar a gravação de áudio
TfLiteStatus InitAudioRecording(tflite::ErrorReporter *error_reporter) {
  // Preenche o buffer de referência com valores de teste ou padrão
  for (size_t i = 0; i < bufferLen; ++i) {
    g_reference_buffer[i] = i % 100; // Exemplo de dados de teste
  }

  // Cria uma tarefa dedicada para capturar amostras de áudio
  xTaskCreatePinnedToCore(CaptureSamples, "CaptureSamples", 4096, NULL, 5, NULL, 1);
  // Aguarda até que o timestamp do áudio seja atualizado
  while (!g_latest_audio_timestamp) {
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  // Log de inicialização bem-sucedida
  ESP_LOGI(TAG, "Gravação de áudio iniciada");
  return kTfLiteOk;
}

static void SendAudioDataViaSerial() {
  // Envia o tamanho do buffer para referência
  Serial.println("Enviando dados do áudio via Serial:");
  for (size_t i = 0; i < bufferLen; ++i) {
    // Envia cada valor do buffer de saída
    Serial.println(g_audio_output_buffer[i]);
    log_d("g_audio_output_buffer: %i", g_audio_output_buffer[i]);
  }
  Serial.println("Envio concluído.");
  log_d("Envio concluído.");
}

// Função para obter as amostras de áudio
TfLiteStatus GetAudioSamples(tflite::ErrorReporter *error_reporter,
                             int start_ms, int duration_ms,
                             int *audio_samples_size, int16_t **audio_samples) {
  // Inicializa a gravação de áudio se ainda não foi feita
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }

  // Retorna o tamanho do buffer e o ponteiro para as amostras
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;

  //SendAudioDataViaSerial();

  return kTfLiteOk;
}

// Função para obter o timestamp mais recente do áudio
int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
