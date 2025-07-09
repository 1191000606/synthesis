import json
import random

import numpy as np
from prompt import get_scale_prompt
from llm import llm_generate
from parse import parse_scale

partnet_category_embeddings = None
partnet_category_list = None
partnet_ids_dict = None
sentence_bert_model = None

def adjust_size(configs):
    object_scales = []
    task_description = ""

    for config in configs:
        if "task_description" in config:
            task_description = config["task_description"]
            continue

        if "name" in config and "size" in config:
            object_scales.append([config["name"].lower(), config["size"]])

    adjust_size_prompt = get_scale_prompt(task_description, object_scales)

    adjust_size_response = llm_generate(adjust_size_prompt)

    corrected_names, corrected_sizes = parse_scale(adjust_size_response)

    for config in configs:
        if "name" in config and "size" in config:
            if config["name"].lower() in corrected_names:
                index = corrected_names.index(config["name"].lower())
                config["size"] = corrected_sizes[index]

    return configs

def match_similar_object_from_partnet(distractor_config):
    global partnet_category_embeddings
    global partnet_category_list
    global partnet_ids_dict
    global sentence_bert_model

    from sentence_transformers import SentenceTransformer, util

    if partnet_category_embeddings is None:
        partnet_category_embeddings = np.load("../data/partnet_category_embeddings.npy")
    
    if partnet_category_list is None:
        with open("../data/partnet_category.txt", "r") as f:
            partnet_category_list = [line.strip() for line in f.readlines()]
    
    if partnet_ids_dict is None:
        with open("../data/partnet_mobility_dict.json", "r") as f:
            partnet_ids_dict = json.load(f)

    if sentence_bert_model is None:
        sentence_bert_model = SentenceTransformer('all-mpnet-base-v2') # 自动根据cuda是否可用选择设备

    for obj in distractor_config:
        obj_name_embeddings = sentence_bert_model.encode(obj['name'])
        similarity = util.cos_sim(obj_name_embeddings, partnet_category_embeddings).cpu().numpy()
        
        if np.max(similarity) > 0.85: # 0.95是不是太高了？
            best_category = partnet_category_list[np.argmax(similarity)]
            obj['type'] = 'urdf'
            obj['name'] = best_category
            obj['reward_asset_path'] = random.choice(partnet_ids_dict[best_category])

    return distractor_config
