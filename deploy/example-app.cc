#include <cstdio>
#include <cstdlib>
#include <vector>
#include <torch/script.h>

std::vector<float> load_audio(const char *audio_path, bool offset_44 = true) {
  FILE *fp = fopen(audio_path, "r");
  int offset = offset_44 ? 44 : 0;
  fseek(fp, 0, SEEK_END);
  int size = ftell(fp) - offset;
  char *buf = (char *)malloc(size);
  fseek(fp, offset, SEEK_SET);
  fread(buf, sizeof(char), size, fp);
  fclose(fp);

  std::vector<float> waveform;
  for (int i = 0; i < size; i += 2) {
    waveform.push_back(*(short *)(buf + i) / 32768.0);
  }

  free(buf);
  return waveform;
}

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <model-path> <wav-path>\n";
    return -1;
  }

  const char *model_path = argv[1];
  const char *wav_path = argv[2];

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // Create a vector of inputs.
  auto waveform = load_audio(wav_path);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::from_blob(waveform.data(), {1, 1, waveform.size()}));

  // Execute the model and turn its output into a tensor.
  auto logits = model.forward(inputs);

  // Greedy search
  auto align = at::squeeze(at::argmax(logits.toTensor(), 2), 1);
  std::vector<long> align_v(align.data_ptr<long>(), align.data_ptr<long>() + align.size(0));

  // The final result
  std::stringstream result_stream;
  char index_to_char[] = "eYESNO ";
  for (size_t i = 0; i < align_v.size(); i++) {
    if (align_v[i] != 0 && (i == 0 || align_v[i] != align_v[i - 1])) {
      result_stream << index_to_char[align_v[i]];
    }
  }
  auto result = result_stream.str();

  std::cout << result << std::endl;
}