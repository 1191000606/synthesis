import json
import random
import yaml
import os
# from prompt import get_joint_angle_prompt, get_spatial_relationship_prompt, get_task_gen_prompt, get_scene_gen_prompt
from prompt.joint_angle import get_joint_angle_prompt
from prompt.spatial_relationship import get_spatial_relationship_prompt
from prompt.task_gen import get_task_gen_prompt
from prompt.scene_gen import get_scene_gen_prompt
from llm import llm_generate,MODEL_NAME, TEMP
from parse import parse_joint_angle_response, parse_response_to_get_yaml, parse_spatial_relationship_response, parse_task_response

# 对于需要的配置信息，可以写到一个配置文件中
# 需要仔细思考每个配置文件是否需要写。对于几乎不会修改的配置，就不用写到配置文件里面

with open("./data/partnet_category.txt", "r") as f:
    partnet_category_list = [line.strip() for line in f.readlines()]

with open("./data/partnet_mobility_dict.json", "r") as f:
    partnet_ids_dict = json.load(f)

object_category = random.choice(partnet_category_list)

object_id = random.choice(partnet_ids_dict[object_category])

object_path = f"../RoboGen/data/dataset/{object_id}"
#print(os.getcwd())###
with open(f"{object_path}/link_and_joint.txt", 'r') as f:
    articulation_tree = f.readlines()

with open(f"{object_path}/semantics.txt", "r") as f:
    semantics = f.readlines()

task_gen_prompt = get_task_gen_prompt(object_category, articulation_tree, semantics)
#print("Task Generation Prompt:")###
#print(task_gen_prompt)###
llm_response = llm_generate(task_gen_prompt)
# print("LLM Response:")###
# print(llm_response)###

task_names, task_descriptions, additional_objects, links, joints = parse_task_response(llm_response)

# print("Parsed Task Names:")###
# print(task_names)###
# print("Parsed Task Descriptions:")###
# print(task_descriptions)###
# print("Parsed Additional Objects:")###
# print(additional_objects)###
# print("Parsed Links:")###
# print(links)###
# print("Parsed Joints:")###
# print(joints)###


# 参照·build_task_given_text

for task_name, task_description, additional_object, link, joint in zip(task_names, task_descriptions, additional_objects, links, joints):
    scene_gen_prompt = get_scene_gen_prompt(task_name, task_description, object_category, additional_object, articulation_tree, semantics)

    response = llm_generate(scene_gen_prompt)
    # print("Scene Generation Response:")###
    # print(response)###

    # parsed_yaml, save_name = parse_response_to_get_yaml(response, task_descriptions, save_path=size_save_path)

    # 试着去掉save_path，这个似乎是记录整个聊天记录的文件的save_path，因此其实是不需要保存的
    parsed_yaml, save_name = parse_response_to_get_yaml(response, task_description)
    # print("Parsed YAML:")###
    # print(parsed_yaml)###   
    # print("Save Name:")###
    # print(save_name)###

    for obj in parsed_yaml:
        if "name" in obj and obj["name"] == object_category:
            obj["type"] = "urdf"
            obj["reward_asset_path"] = object_path

    joint_angle_prompt = get_joint_angle_prompt(task_name, task_description, articulation_tree, semantics, links, joints)

    response = llm_generate(joint_angle_prompt)

    joint_angle_values = parse_joint_angle_response(response)

    joint_angle_values["set_joint_angle_object_name"] = object_category #

    spatial_relationships_prompt = get_spatial_relationship_prompt(task_name, task_description, additional_object, articulation_tree, semantics, links, joints)

    response = llm_generate(spatial_relationships_prompt)

    spatial_relationships = parse_spatial_relationship_response(response)#
    
    
    # parsed_yaml.append(dict(solution_path=solution_path)) 这个还没搞
    
    parsed_yaml.append(joint_angle_values)
    parsed_yaml.append(dict(spatial_relationships=spatial_relationships))
    parsed_yaml.append(dict(task_name=task_name, task_description=task_description))
    
    print("Config:")###
    print(parsed_yaml)###
    

    # Todo: 后面还需要补充distractor的生成

    # 中间不要有太多的额外的代码，最后再一起把生成的yaml写到文件里面

# Todo: 大模型生成相关的内容写到llm.py里面，凡是与具体任务无关的配置，其实都应该写到llm.py里面
