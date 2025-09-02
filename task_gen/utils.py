import json
import pickle
import random

import numpy as np
from FlagEmbedding import BGEM3FlagModel
import pandas
from sentence_transformers import SentenceTransformer, util
import objaverse

from prompt import get_scale_prompt
from llm import llm_generate
from parse import parse_scale

# for partnet object matching
partnet_category_embeddings = None
partnet_category_list = None
partnet_ids_dict = None
sentence_bert_model = None

# for objaverse object matching
objaverse_object_embeddings = None
objaverse_bge_m3_model = None
objaverse_uids = None
objaverse_xl_dataframe = None

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

    if partnet_category_embeddings is None:
        partnet_category_embeddings = np.load("./data/partnet/category_embeddings.npy")

    if partnet_category_list is None:
        with open("./data/partnet/category.txt", "r") as f:
            partnet_category_list = [line.strip() for line in f.readlines()]

    if partnet_ids_dict is None:
        with open("./data/partnet/category_to_ids.json", "r") as f:
            partnet_ids_dict = json.load(f)

    if sentence_bert_model is None:
        sentence_bert_model = SentenceTransformer('all-mpnet-base-v2') # 自动根据cuda是否可用选择设备

    for obj in distractor_config:
        obj_name_embeddings = sentence_bert_model.encode(obj['name'])

        # Todo：这里的余弦相似度是否有必要用util，后续可以改成自己实现的
        similarity = util.cos_sim(obj_name_embeddings, partnet_category_embeddings).cpu().numpy()

        if np.max(similarity) > 0.85: # 0.95是不是太高了？
            best_category = partnet_category_list[np.argmax(similarity)]
            obj['type'] = 'urdf'
            obj['name'] = best_category
            obj['asset_path'] = "./data/partnet/dataset/" + str(random.choice(partnet_ids_dict[best_category]))

    return distractor_config


def match_similar_object_from_objaverse(distractor_config):
    for obj in distractor_config:
        if "type" in obj and obj["type"] == "urdf":
            continue

        obj_description = obj["lang"]

        global objaverse_object_embeddings
        global objaverse_bge_m3_model
        global objaverse_uids
        global objaverse_xl_dataframe

        if objaverse_object_embeddings is None:
            objaverse_uids, objaverse_object_embeddings = pickle.load(open("./data/objaverse/cap3d_full_bgem3_embeddings.pkl", "rb"))

        if objaverse_bge_m3_model is None:    
            objaverse_bge_m3_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

        obj_description_embedding = objaverse_bge_m3_model.encode([obj_description], max_length=512)['dense_vecs'][0]

        similarity = util.cos_sim(obj_description_embedding, objaverse_object_embeddings).cpu().numpy()[0]

        best_index = np.argmax(similarity)
        best_score = similarity[best_index]

        if best_score < 0.8:
            continue

        best_uid = objaverse_uids[best_index]

        if len(best_uid) == 32: # Objaverse 1.0
            objaverse.load_objects([best_uid], download_processes=1)
        elif len(best_uid) == 64: # Objaverse-XL
            if objaverse_xl_dataframe is None:
                objaverse_xl_dataframe = pandas.read_csv("./data/objaverse/objaverse_xl.csv")

            dataframe = objaverse_xl_dataframe[objaverse_xl_dataframe['sha256'] == best_uid]
            objaverse.xl.download_objects(dataframe, download_processes=1)
        else:
            assert False, f"Invalid Objaverse UID '{best_uid}'"

        obj['asset_path'] = f"./data/objaverse/dataset/{best_uid}"

    return distractor_config
