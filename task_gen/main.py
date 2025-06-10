import json
import random
import yaml
import os
import time,datetime
# from prompt import get_joint_angle_prompt, get_spatial_relationship_prompt, get_task_gen_prompt, get_scene_gen_prompt
from prompt.joint_angle import get_joint_angle_prompt
from prompt.spatial_relationship import get_spatial_relationship_prompt
from prompt.task_gen import get_task_gen_prompt
from prompt.scene_gen import get_scene_gen_prompt
from llm import llm_generate
from parse import parse_joint_angle_response, parse_response_to_get_yaml, parse_spatial_relationship_response, parse_task_response
from prompt.prompt_distractor import generate_distractor


# 对于需要的配置信息，可以写到一个配置文件中
# 需要仔细思考每个配置文件是否需要写。对于几乎不会修改的配置，就不用写到配置文件里面

with open("./data/partnet_category.txt", "r") as f:
    partnet_category_list = [line.strip() for line in f.readlines()]

with open("./data/partnet_mobility_dict.json", "r") as f:
    partnet_ids_dict = json.load(f)

object_category = random.choice(partnet_category_list)

object_id = random.choice(partnet_ids_dict[object_category])

# object_path = f"../RoboGen/data/dataset/{object_id}"  软链接前
object_path = f"data/dataset/{object_id}"

with open(f"{object_path}/link_and_joint.txt", 'r') as f:
    articulation_tree = f.readlines()

with open(f"{object_path}/semantics.txt", "r") as f:
    semantics = f.readlines()

task_gen_prompt = get_task_gen_prompt(object_category, articulation_tree, semantics)

llm_response = llm_generate(task_gen_prompt)

task_names, task_descriptions, additional_objects, links, joints = parse_task_response(llm_response)




# 参照·build_task_given_text

for task_name, task_description, additional_object, link, joint in zip(task_names, task_descriptions, additional_objects, links, joints):
    scene_gen_prompt = get_scene_gen_prompt(task_name, task_description, object_category, additional_object, articulation_tree, semantics)

    response = llm_generate(scene_gen_prompt)

    parsed_yaml, save_name = parse_response_to_get_yaml(response, task_description)


    for obj in parsed_yaml:
        if "name" in obj and obj["name"] == object_category:
            obj["type"] = "urdf"
            obj["reward_asset_path"] = object_path

    joint_angle_prompt = get_joint_angle_prompt(task_name, task_description, articulation_tree, semantics, link, joint)

    response = llm_generate(joint_angle_prompt)

    joint_angle_values = parse_joint_angle_response(response)

    joint_angle_values["set_joint_angle_object_name"] = object_category #

    spatial_relationships_prompt = get_spatial_relationship_prompt(task_name, task_description, additional_object, articulation_tree, semantics, link, joint)

    response = llm_generate(spatial_relationships_prompt)

    spatial_relationships = parse_spatial_relationship_response(response)#
    
    
    # parsed_yaml.append(dict(solution_path=solution_path)) 这个还没搞，其实就是本yaml文件的路径,好像也没用
    
    parsed_yaml.append(joint_angle_values)
    parsed_yaml.append(dict(spatial_relationships=spatial_relationships))
    parsed_yaml.append(dict(task_name=task_name, task_description=task_description))
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    meta_path="generated_tasks_release"
    # save_folder = "data/{}/{}_{}_{}".format(meta_path, object_category, object_path, time_string)
    save_folder = "data/{}/{}_{}_{}".format(meta_path, object_category, object_id, time_string)
    
    
    config_path = os.path.join(save_folder, save_name)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(parsed_yaml, f, indent=4)
    
    
    
    
    
    
    print("Config:")###
    print(parsed_yaml)###
    generate_distractor(config_path)
    

    # Todo: 后面还需要补充distractor的生成

    # 中间不要有太多的额外的代码，最后再一起把生成的yaml写到文件里面

# Todo: 大模型生成相关的内容写到llm.py里面，凡是与具体任务无关的配置，其实都应该写到llm.py里面
