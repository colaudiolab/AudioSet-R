# -*- coding: utf-8 -*-

import config
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import string
import pandas as pd
import numpy as np
import requests

from openai import OpenAI

import laion_clap

import re

def load_CLAP(pretrained_path):
    CLAP_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    CLAP_model.load_ckpt(pretrained_path)
    return CLAP_model

def cal_cos_sim(model, audio_path, input_text_data_list):
    cos_sim_list = []
    ### Directly get audio embeddings from audio files
    audio_file = [audio_path]
    audio_emb = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
    audio_emb = audio_emb.reshape(-1)

    text_embed_list = model.get_text_embedding(input_text_data_list)
    for text_emb in text_embed_list:
        cos_sim = audio_emb.dot(text_emb)/(np.linalg.norm(audio_emb)*np.linalg.norm(text_emb))
        cos_sim_list.append(cos_sim)
    
    return cos_sim_list  

def parse_matched_labels(match_content, label_set):
    text = match_content.strip()
    words = [w.strip() for w in text.split(',')]  
    result = []

    i = 0
    while i < len(words):
        matched = None
        for j in range(len(words), i, -1):
            candidate = ', '.join(words[i:j])
            if candidate in label_set:
                matched = candidate
                i = j
                break
        if matched:
            result.append(matched)
        else:
            
            result.append(words[i])
            i += 1
    return result

def pad_str(str_list, pad_mode):
    
    if len(str_list) == 1:
        return str_list
        
    if pad_mode == "min":
        min_len = min(len(s) for s in str_list)
        new_str_list = [s[:min_len] for s in str_list]
        
    elif pad_mode=="max":
        max_length = max(len(s) for s in str_list)
        new_str_list = [s + s * ((max_length - len(s)) // len(s)) for s in str_list]
    return new_str_list   

def pad_as_AC_str(str_list):
    AC_str = str_list[-1]
    GPT_str = int(np.ceil(len(str_list[0])/len(AC_str)))*AC_str
    Mistral_str = int(np.ceil(len(str_list[1])/len(AC_str)))*AC_str
    str_list[0] = GPT_str
    str_list[1] = Mistral_str
    
    return str_list

def pad_as_lb_str(str_list):
    lb_str = str_list[-1]+'.'
    GPT_str = int(np.ceil(len(str_list[0])/len(lb_str)))*lb_str
    Mistral_str = int(np.ceil(len(str_list[1])/len(lb_str)))*lb_str
    AC_str = int(np.ceil(len(str_list[2])/len(lb_str)))*lb_str
    str_list[0] = GPT_str
    str_list[1] = Mistral_str
    str_list[2] = AC_str
    
    return str_list      
     

def remove_punctuation(input_string):

    translator = str.maketrans(string.punctuation, " "*len(string.punctuation))
    result_string = input_string.translate(translator)
    
    return result_string

def LLMs_main(config):

    print('######## LLMs Predictive Labeling! ########')    
    
    as_csv = pd.read_csv(config.as_csv_path, sep=',')
    label_csv = pd.read_csv(config.label_csv_path, sep=',')    
    
    display_name_series = label_csv['display_name']
    display_name_list = display_name_series.tolist()
    
    with open(config.default_prompt_path, 'r') as log_f:
        log_lines = log_f.read()
    default_prompt = log_lines
    
    with open(config.labels_prompt_path, 'r', encoding='gbk') as log_f:
        log_lines = log_f.read()
    labels_prompt = log_lines
    
    speech_keyword = ["no speech", "does not", "is not"]
    music_keyword = ["no music", "does not", "is not"]
    
    CLAP_model = load_CLAP(config.pretrained_path)
    print('######## Loading CLAP model! ########') 
    
    filename_list = []
    label_list = []
    CLAP_label_score = []
    CLAP_Mistral_score = []
    content_list = []
    
    for index, row in as_csv.iterrows(): 
        
        filename = row['filename']
        wav_path = os.path.join(config.root_audio_path, filename)   
        #! label = row['label']
        
        qwen_caption_save_path = os.path.join(config.qwen_caption_path, filename[:-4]+'.txt')
        os.makedirs(config.mis_caption_path, exist_ok=True)
        mis_caption_save_path = os.path.join(config.mis_caption_path, filename[:-4]+'.txt')
        
        if os.path.exists(qwen_caption_save_path) and not os.path.exists(mis_caption_save_path): #
                print(index, filename)
            # try:
                for i in range(config.mistral_try_num):
                    best_cap_score = []
                    best_cap = ''
                    
                    with open(qwen_caption_save_path, 'r') as c_f:   
                        qwen_caption = c_f.readlines()
    
                    speech_caption = qwen_caption[1]
                    music_caption = qwen_caption[2]
                    
                    if any(keyword in speech_caption for keyword in speech_keyword):
                        speech_caption = ""
                    if any(keyword in music_caption for keyword in music_keyword):
                        music_caption = ""
                    qwen_prompt = 'Details:\n1 Crowd-sourced workers:{'+'{}{}{}'.format(qwen_caption[0], speech_caption, music_caption)+'}'
                    qwen_prompt = qwen_prompt+"My instructions:\n\
                        Do not mention the specific content of speech! Do not mention the specific content of speech!\
                        \nOutput your caption (Within 50 words):"
                    
                    request_data = {
                        "model": "Mistral",
                        "messages": [{"role": "system", "content": default_prompt}, 
                                      {"role": "user", "content": qwen_prompt}],
                        "stream": False
                    }

                    
                    url = "http://localhost:11434/api/chat"
                    response = requests.post(url, json=request_data)
                    
                    response_data = response.json()
                    content = response_data.get("message", {}).get("content")
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                    content = content.strip()
                    print("content:", content)
                  
                    display_name_list = set(display_name_list)  
                    content = parse_matched_labels(content, display_name_list)
                    
                    label_string = ", ".join(display_name_list)
                    match_prompt = 'Here is a list of predicted audio labels:{'+'{}'.format(content)+'}'
                    match_prompt = match_prompt+'\n'+'Compare these with the following official AudioSet labels:{'+'{}'.format(label_string)+'}'
                    match_prompt = match_prompt+'\n'+'For each predicted tag, find the most semantically relevant AudioSet label. You can match based on exact string or common synonyms. Ensure every selected label comes directly from the AudioSet list.'
                    match_prompt = match_prompt+'\n'+'Output only the final matched AudioSet labels, separated by commas. Do not include explanations, line breaks, or any other text.'
                    
                    client = OpenAI(
                        api_key="your_api_key",           
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                    
                    completion = client.chat.completions.create(
                        model="deepseek-r1",  
                        messages=[
                            {"role": "system", "content": labels_prompt},
                            {"role": "user", "content": match_prompt},
                        ]
                    )
                    print(completion.choices[0].message.content)
                    match_content = completion.choices[0].message.content
                    match_content = match_content.strip()
                    
                    #input_text = match_content.split(',')
                    display_name_list = set(display_name_list)  
                    input_text = parse_matched_labels(match_content, display_name_list)

                    print("Predictive labels:",input_text)
                
                    mis_cos_sim = cal_cos_sim(CLAP_model, wav_path, input_text)

                    if i == 0 or mis_cos_sim >= best_cap_score:
                        best_cap_score = mis_cos_sim
                        best_cap = match_content
                
                with open(mis_caption_save_path, 'w') as out_f:
                    out_f.write(best_cap)
                
                filename_list.append(filename)
                CLAP_Mistral_score.append(best_cap_score)
                content_list.append(input_text)
                
                CLAP_score_df = pd.DataFrame({'filename':filename_list, "Mistral_score":CLAP_Mistral_score, "Predictive labels":content_list})
                CLAP_score_df.to_csv(config.CLAP_score_csv_path, index=False) 
            
            
if __name__ == "__main__":
    
    LLMs_main(config)
