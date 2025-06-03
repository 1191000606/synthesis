import json
import random

from prompt import get_joint_angle_prompt, get_spatial_relationship_prompt, get_task_gen_prompt, get_scene_gen_prompt
from llm import llm_generate
from parse import parse_joint_angle_response, parse_response_to_get_yaml, parse_spatial_relationship_response, parse_task_response

with open("./data/partnet_category.txt", "r") as f:
    partnet_category_list = [line.strip() for line in f.readlines()]

with open("./data/partnet_mobility_dict.json", "r") as f:
    partnet_ids_dict = json.load(f)

object_category = random.choice(partnet_category_list)

object_id = random.choice(partnet_ids_dict[object_category])

object_path = f"./data/dataset/{object_id}"

with open(f"{object_path}/link_and_joint.txt", 'r') as f:
    articulation_tree = f.readlines()

with open(f"{object_path}/semantics.txt", "r") as f:
    semantics = f.readlines()

task_gen_prompt = get_task_gen_prompt(object_category, "".join(articulation_tree), "".join(semantics))

llm_response = llm_generate(task_gen_prompt)

task_names, task_descriptions, additional_objects, links, joints = parse_task_response(llm_response)

for task_name, task_description, additional_object, link, joint in zip(task_names, task_descriptions, additional_objects, links, joints):
    scene_gen_prompt = get_scene_gen_prompt(task_name, task_description, object_category, additional_object, articulation_tree, semantics)

    response = llm_generate(scene_gen_prompt)

    parsed_yaml, save_name = parse_response_to_get_yaml(task_yaml_response, description, save_path=size_save_path, temperature=temperature_dict["size"], model=model_dict["size"])

    for obj in parsed_yaml:
        if "name" in obj and obj["name"] == object_category:
            obj["type"] = "urdf"
            obj["reward_asset_path"] = object_path

    joint_angle_prompt = get_joint_angle_prompt(task_name, task_description, articulation_tree, semantics, links, joints)

    response = llm_generate(joint_angle_prompt)

    joint_angle_values = parse_joint_angle_response(response)

    joint_angle_values["set_joint_angle_object_name"] = object_category

    spatial_relationships_prompt = get_spatial_relationship_prompt(task_name, task_description, additional_object, articulation_tree, semantics, links, joints)

    response = llm_generate(spatial_relationships_prompt)

    spatial_relationships = parse_spatial_relationship_response()
