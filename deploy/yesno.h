#pragma once

#include <string>
#include <vector>

void *load_model(const char *model_path);
void delete_model(void *model);
std::string recognize(void *model, const std::vector<float> &waveform);
std::vector<float> convert_audio_from_int16_to_fp32(const char *buf, const int size);
