#include <cstdio>
#include "yesno.h"

const int MAX_BUF_SIZE = 128 * 1024;

int read_file(const char *file, char *buf, bool offset_44 = true) {
  FILE *fp = fopen(file, "r");
  int offset = offset_44 ? 44 : 0;
  fseek(fp, 0, SEEK_END);
  int bytes_num = ftell(fp) - offset;
  if (bytes_num > MAX_BUF_SIZE) return 0;
  fseek(fp, offset, SEEK_SET);
  bytes_num = fread(buf, sizeof(char), bytes_num, fp);
  fclose(fp);
  return bytes_num;
}

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    puts("usage: example-app <model-path> <wav-path>");
    return -1;
  }
  const char *model_path = argv[1];
  const char *wav_path = argv[2];

  // load audio
  char wav_buf[MAX_BUF_SIZE];
  int wav_size = read_file(wav_path, wav_buf);
  auto waveform = convert_audio_from_int16_to_fp32(wav_buf, wav_size);

  // load model
  void *model = load_model(model_path);

  // recognize
  std::string result = recognize(model, waveform);
  puts(result.c_str());

  delete_model(model);
}