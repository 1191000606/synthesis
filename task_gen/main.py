import datetime
import json
import os
import random
import time
import yaml

from prompt import get_task_gen_prompt, get_scene_gen_prompt, get_joint_angle_prompt, get_spatial_relationship_prompt, get_distractor_prompt, get_scale_prompt
from parse import parse_joint_angle_response, parse_response_to_get_yaml, parse_scale, parse_spatial_relationship_response, parse_task_response
from llm import llm_generate
from utils import retrieve_object_from_partnet, retrieve_object_from_objaverse

with open("./data/partnet/category.txt", "r") as f:
    partnet_category_list = [line.strip() for line in f.readlines()]

with open("./data/partnet/category_to_ids.json", "r") as f:
    partnet_ids_dict = json.load(f)

object_category = random.choice(partnet_category_list)

object_id = random.choice(partnet_ids_dict[object_category])

object_path = f"./data/partnet/dataset/{object_id}"

with open(f"{object_path}/link_and_joint.txt", 'r') as f:
    articulation_tree = f.readlines()
    articulation_tree = "".join(articulation_tree)[:-1] # 去掉最后的换行符

with open(f"{object_path}/semantics.txt", "r") as f:
    semantics = f.readlines()
    semantics = "".join(semantics)[:-1] # 去掉最后的换行符

task_gen_prompt = get_task_gen_prompt(object_category, articulation_tree, semantics)
task_gen_response = llm_generate(task_gen_prompt)
task_attrs = parse_task_response(task_gen_response)

for task_name, task_description, additional_object, link, joint in zip(*task_attrs):
    task_attr = {
        "task_name": task_name,
        "task_description": task_description,
        "object_category": object_category,
        "additional_object": additional_object,
        "articulation_tree": articulation_tree,
        "semantics": semantics,
        "links": link,
        "joints": joint,
    }

    configs = [{
        "task_name": task_name,
        "task_description": task_description,
    }]

    scene_gen_prompt = get_scene_gen_prompt(task_attr)
    scene_gen_response = llm_generate(scene_gen_prompt)
    scene_config = parse_response_to_get_yaml(scene_gen_response)

    for obj in scene_config:
        if "name" in obj and obj["name"] == object_category:
            obj["type"] = "urdf"
            obj["asset_path"] = object_path

    configs.extend(scene_config)

    adjust_size_prompt = get_scale_prompt(configs)
    adjust_size_response = llm_generate(adjust_size_prompt)
    corrected_names, corrected_sizes = parse_scale(adjust_size_response)

    for config in configs:
        if "name" in config and "size" in config:
            if config["name"].lower() in corrected_names:
                index = corrected_names.index(config["name"].lower())
                config["size"] = corrected_sizes[index]

    joint_angle_prompt = get_joint_angle_prompt(task_attr)
    joint_angle_response = llm_generate(joint_angle_prompt)
    joint_angle_config = parse_joint_angle_response(joint_angle_response)

    joint_angle_config["set_joint_angle_object_name"] = object_category
    configs.append(joint_angle_config)

    spatial_relationships_prompt = get_spatial_relationship_prompt(task_attr)
    spatial_relationships_response = llm_generate(spatial_relationships_prompt)
    spatial_relationships_config = parse_spatial_relationship_response(spatial_relationships_response)

    configs.append(spatial_relationships_config)

    distractor_prompt = get_distractor_prompt(task_name, task_description, scene_config)
    distractor_response = llm_generate(distractor_prompt)
    distractor_config = parse_response_to_get_yaml(distractor_response)

    distractor_config = retrieve_object_from_partnet(distractor_config)

    distractor_config = retrieve_object_from_objaverse(distractor_config)

    time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S') # Todo：这里可能后续会有冲突，比如高并发的时候，到时候可以加上一个随机数或者UUID
    save_folder = f"./data/task_config/{object_category}_{object_id}_{time_string}"

    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, "scene.yaml"), 'w') as f:
        yaml.dump(configs, f, indent=4)

    with open(os.path.join(save_folder, "distractor.yaml"), 'w') as f:
        yaml.dump(distractor_config, f, indent=4)

    print(f"Generated task for '{task_name}' saved in {save_folder}")
