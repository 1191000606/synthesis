import datetime
import json
import os
import random
import time

from prompt import get_task_gen_prompt, get_scene_gen_prompt, get_joint_angle_prompt, get_spatial_relationship_prompt, get_distractor_prompt
from parse import parse_joint_angle_response, parse_response_to_get_yaml, parse_spatial_relationship_response, parse_task_response

from llm import llm_generate
from utils import adjust_size, match_similar_object_from_partnet

with open("../data/partnet_category.txt", "r") as f:
    partnet_category_list = [line.strip() for line in f.readlines()]

with open("../data/partnet_mobility_dict.json", "r") as f:
    partnet_ids_dict = json.load(f)

object_category = random.choice(partnet_category_list)

object_id = random.choice(partnet_ids_dict[object_category])

# object_path = f"../RoboGen/data/dataset/{object_id}"  软链接前
object_path = f"../data/dataset/{object_id}"

with open(f"{object_path}/link_and_joint.txt", 'r') as f:
    articulation_tree = f.readlines()
    articulation_tree = "".join(articulation_tree)[:-1]

with open(f"{object_path}/semantics.txt", "r") as f:
    semantics = f.readlines()
    semantics = "".join(semantics)[:-1]

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
            obj["reward_asset_path"] = object_path

    configs.extend(scene_config)

    adjust_size(configs)

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

    distractor_config = match_similar_object_from_partnet(distractor_config)
    
    time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    save_folder = f"../data/generated_tasks_release/{object_category}_{object_id}_{time_string}"
    
    os.makedirs(save_folder, exist_ok=True)
    
    with open(os.path.join(save_folder, "scene.json"), 'w') as f:
        json.dump(configs, f, indent=4)
    
    with open(os.path.join(save_folder, "distractor.json"), 'w') as f:
        json.dump(distractor_config, f, indent=4)

    print(f"Generated task for '{task_name}' saved in {save_folder}")
