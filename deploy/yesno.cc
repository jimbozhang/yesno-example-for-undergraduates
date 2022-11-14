#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <torch/script.h>
#include "yesno.h"

void *load_model(const char *model_path) {
  torch::jit::script::Module *model = new torch::jit::script::Module;
  try {
    *model = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return nullptr;
  }
  return (void *)model;
}

void delete_model(void *model) {
  delete (torch::jit::script::Module *)model;
}

std::string recognize(void *model_v, const std::vector<float> &waveform) {
  // Create a vector of inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<float> input = waveform;
  inputs.push_back(torch::from_blob(input.data(), {1, 1, input.size()}));

  // Execute the model
  auto *model = (torch::jit::script::Module *)model_v;
  auto logits = model->forward(inputs);

  // Greedy search
  auto align = at::squeeze(at::argmax(logits.toTensor(), 2), 1);
  std::vector<long> align_v(align.data_ptr<long>(), align.data_ptr<long>() + align.size(0));

  // The final result
  std::stringstream result;
  char index_to_char[] = "eYESNO ";
  for (size_t i = 0; i < align_v.size(); i++) {
    if (align_v[i] != 0 && (i == 0 || align_v[i] != align_v[i - 1])) {
      result << index_to_char[align_v[i]];
    }
  }
  return result.str();
}

std::vector<float> convert_audio_from_int16_to_fp32(const char *buf, const int size) {
  std::vector<float> waveform;
  for (int i = 0; i < size; i += sizeof(short))
    waveform.push_back(*(short *)(buf + i) / float(SHRT_MAX));
  return waveform;
}
