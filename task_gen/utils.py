import json
import os
import pickle
import random
import shutil

import numpy as np
from FlagEmbedding import BGEM3FlagModel
import pandas
from sentence_transformers import SentenceTransformer, util
import objaverse
import objaverse.xl
import trimesh
import pybullet as p

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

def retrieve_object_from_partnet(distractor_config):
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
        sentence_bert_model = SentenceTransformer("all-mpnet-base-v2") # 自动根据cuda是否可用选择设备

    for obj in distractor_config:
        obj_name_embeddings = sentence_bert_model.encode(obj["name"])

        # Todo：这里的余弦相似度是否有必要用util，后续可以改成自己实现的
        similarity = util.cos_sim(obj_name_embeddings, partnet_category_embeddings).cpu().numpy()

        if np.max(similarity) > 0.85:  # 0.95是不是太高了？
            best_category = partnet_category_list[np.argmax(similarity)]
            obj["type"] = "urdf"
            obj["name"] = best_category
            obj["asset_path"] = "./data/partnet/dataset/" + str(random.choice(partnet_ids_dict[best_category]))

    return distractor_config


def retrieve_object_from_objaverse(distractor_config):
    for obj in distractor_config:
        if "type" in obj and obj["type"] == "urdf":
            continue

        obj_description = obj["lang"]

        global objaverse_object_embeddings
        global objaverse_bge_m3_model
        global objaverse_uids

        if objaverse_object_embeddings is None:
            objaverse_uids, objaverse_object_embeddings = pickle.load(open("./data/objaverse/cap3d_full_bgem3_embeddings.pkl", "rb"))

        if objaverse_bge_m3_model is None:
            objaverse_bge_m3_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

        obj_description_embedding = objaverse_bge_m3_model.encode([obj_description], max_length=512)["dense_vecs"][0]

        similarity = util.cos_sim(obj_description_embedding, objaverse_object_embeddings).cpu().numpy()[0]

        best_index = np.argmax(similarity)
        best_score = similarity[best_index]

        if best_score < 0.8:
            continue

        best_uid = objaverse_uids[best_index]

        download_path = download_objaverse_object(best_uid)

        process_success = process_objaverse_object(best_uid, download_path)

        if process_success:
            obj["asset_path"] = f"./data/objaverse/dataset/{best_uid}"

    return distractor_config


def handle_found_object(local_path, file_identifier, sha256, metadata):
    save_dir = f"{os.path.expanduser('~')}/.objaverse/xl_dataset/"
    os.makedirs(save_dir, exist_ok=True)

    shutil.move(local_path, f"{save_dir}/{sha256}.{file_identifier.split('.')[-1]}")


def download_objaverse_object(uid):
    if len(uid) == 32: # Objaverse 1.0
        download_path = objaverse.load_objects([uid], download_processes=1)[uid]
    elif len(uid) == 64: # Objaverse-XL
        global objaverse_xl_dataframe

        if objaverse_xl_dataframe is None:
            objaverse_xl_dataframe = pandas.read_csv("./data/objaverse/objaverse_xl.csv")
            
            # 避免github仓库下载过程中出现交互式提示
            env = os.environ.copy()
            env['GIT_TERMINAL_PROMPT'] = '0'

        dataframe = objaverse_xl_dataframe[objaverse_xl_dataframe['sha256'] == uid]

        objaverse.xl.download_objects(dataframe, download_processes=1, handle_found_object=handle_found_object)

        download_path = f"{os.path.expanduser('~')}/.objaverse/xl_dataset/{uid}.{dataframe['fileType'].iloc[0]}"
    else:
        assert False, f"Invalid Objaverse UID '{uid}'"

    return download_path


def process_objaverse_object(uid, download_path):
    if not os.path.exists(download_path):
        return False

    scene = trimesh.load(download_path)

    obj_data_dir = f"./data/objaverse/dataset/{uid}"

    os.makedirs(obj_data_dir, exist_ok=True)

    trimesh.exchange.export.export_mesh(scene, f"{obj_data_dir}/material.obj")

    normalize_obj(f"{obj_data_dir}/material.obj", f"{obj_data_dir}/material_normalized.obj")

    p.connect(p.DIRECT)

    p.vhacd(f"{obj_data_dir}/material_normalized.obj", f"{obj_data_dir}/material_normalized_vhacd.obj", f"{obj_data_dir}/log.txt")

    obj_to_urdf(obj_data_dir, scale=1)

    return True


def normalize_obj(src_obj_file_path, dst_obj_file_path):
    vertices = []
    with open(src_obj_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:]])

    vertices = np.array(vertices).reshape(-1, 3)
    vertices = vertices - np.mean(vertices, axis=0)  # center to zero
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1))  # normalize to -1, 1

    with open(dst_obj_file_path, "w") as f:
        vertex_idx = 0
        for line in lines:
            if line.startswith("v "):
                line = "v " + " ".join([str(x) for x in vertices[vertex_idx]]) + "\n"
                vertex_idx += 1
            f.write(line)


def obj_to_urdf(obj_data_dir, scale=1):
    all_files = os.listdir(obj_data_dir)

    # 这里有问题，可能会有多个png文件，多个PNG文件一起放入texture，但只有一个obj，这就会报错。后续如果要解决这个问题，可能需要把obj文件拆分
    # 或者是不转为urdf，直接用obj文件
    png_file = None
    for x in all_files:
        if x.endswith(".png"):
            png_file = x
            break

    if png_file is not None:
        material = f"""
      <material name="texture">
        <texture filename="{obj_data_dir}/{png_file}"/>
      </material>"""
    else:
        # Todo: 后面颜色也需要注意
        material = """
      <material name="yellow">
        <color rgba="1 1 0.4 1"/>
      </material>
        """

    # 这里的面cube.urdf要留心
    text = f"""<?xml version="1.0" ?>
<robot name="cube.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>

    <inertial>
      <origin rpy="0 0 0" xyz="0.0 0.02 0.0"/>
      <mass value=".1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>

    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{obj_data_dir}/material_normalized.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      {material}
    </visual>

    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{obj_data_dir}/material_normalized_vhacd.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
  </link>
</robot>
  """

    with open(f"{obj_data_dir}/material.urdf", "w") as f:
        f.write(text)
