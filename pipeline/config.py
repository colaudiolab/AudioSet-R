# -*- coding: utf-8 -*-

### general
gpu_id = '0'
root_audio_path = r"./audioset/" # root path for audio clips
as_csv_path = r"./filename.csv" # csv path including filename

### Audioset label
label_csv_path = r'./class_labels_indices.csv'  # csv path including Audioset label index and display_name(527)

### Qwen-audio
qwen_caption_path = r'./qwen/' # path for saving qwen-auido captions
model_id = 'qwen/Qwen-Audio-Chat'
revision = 'master'
cache_dir = r'/Audioset_caption/' # path for downloading qwen-auido ckpts
qwen_try_num = 2

### Mistral and DeepSeek R1
default_prompt_path = r'./default_prompt.txt' # mistral prompt
labels_prompt_path = r'./labels_prompt.txt'  # deepseek prompt
Predictive_labels_path = r'./predictive_labels/' # path for saving Predictive labels

### CLAP
pretrained_path = r'./music_speech_audioset_epoch_15_esc_89.98.pt' # pre-trained CLAP ckpt path
CLAP_score_csv_path = r'./labels_scores.csv' # csv path for saving labels scores from CLAP
