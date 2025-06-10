from llm import llm_generate
import yaml
import re
from prompt.prompt_with_scale import user_contents_v2 as scale_user_contents_v2, assistant_contents_v2 as scale_assistant_contents_v2
import copy

def parse_task_response(task_response):
    task_names = []
    task_descriptions = []
    additional_objects = []
    links = []
    joints = []

    task_response = task_response.split("\n")
    for l_idx, line in enumerate(task_response):
        if line.lower().startswith("task name:"):
            task_name = line.split(":")[1].strip()
            task_names.append(task_name)

            task_description = task_response[l_idx + 1].split(":")[1].strip()
            task_descriptions.append(task_description)

            additional_objects.append(task_response[l_idx + 2].split(":")[1].strip())

            involved_links = ""
            for link_idx in range(l_idx + 4, len(task_response)):
                if task_response[link_idx].lower().startswith("joints:"):
                    break
                else:
                    # involved_links.append(task_response[link_idx].split(":")[0][2:])
                    involved_links += task_response[link_idx][2:]
            links.append(involved_links)

            involved_joints = ""
            for joint_idx in range(link_idx + 1, len(task_response)):
                if not task_response[joint_idx].lower().startswith("- "):
                    break
                else:
                    # involved_joints.append(task_response[joint_idx].split(":")[0][2:])
                    involved_joints += task_response[joint_idx][2:]
            joints.append(involved_joints)

    return task_names, task_descriptions, additional_objects, links, joints

def parse_response_to_get_yaml(response, task_description):
    yaml_string = []
    if isinstance(response, str):
        response = response.splitlines()

    for l_idx, line in enumerate(response):
        if "```yaml" in line:
            for l_idx_2 in range(l_idx + 1, len(response)):
                if response[l_idx_2].lstrip().startswith("```"):
                    break

                yaml_string.append(response[l_idx_2])

            yaml_string = '\n'.join(yaml_string)
            description = f"{task_description}".replace(" ", "_").replace(".", "").replace(",", "").replace("(", "").replace(")", "")
            save_name =  description + '.yaml'

            # print("=" * 30)
            # print("querying GPT to adjust the size of the objects")
            # print("=" * 30)
            parsed_size_yaml = adjust_size_v2(description, yaml_string)

            return parsed_size_yaml, save_name

def parse_center(center):   
    if center.startswith("(") or center.startswith("["):
        center = center[1:-1]

    center = center.split(",")
    center = [float(x) for x in center]
    return np.array(center)


def adjust_size_v2(task_description, yaml_string):
    # extract object names and sizes
    object_names = [] 
    object_sizes = []
    object_types = []

    config = yaml.safe_load(yaml_string)
    for obj in config:
        if "name" in obj:
            object_names.append(obj["name"].lower())
            object_types.append(obj["type"])
            if obj["type"] == "mesh" or obj["type"] == "urdf" or obj["type"] == "sphere":
                object_sizes.append(obj["size"])
            if obj["type"] in ["cylinder", "cube", "box"]:
                if isinstance(obj["size"], list):
                    object_sizes.append([str(x) for x in obj["size"]])
                else:
                    object_sizes.append([str(x) for x in parse_center(obj["size"])])

    new_user_contents = "```\n"
    better_task_description = re.sub(r"\d", "", task_description)
    better_task_description = better_task_description.replace("_", " ")
    better_task_description = better_task_description.lstrip()
    better_task_description = better_task_description.strip()
    new_user_contents += "Task: {}\n".format(better_task_description)
    for name, type, size in zip(object_names, object_types, object_sizes):
        if type in ["mesh", "urdf", "sphere"]:
            new_user_contents += "{}, {}, {}\n".format(name, type, size)
        else:
            new_content = "{}, {}, ".format(name, type)
            size_string = ", ".join(size)
            new_content = new_content + size_string + "\n"
            new_user_contents += new_content
    new_user_contents += "```"
    input_user = copy.deepcopy(scale_user_contents_v2)
    input_user.append(new_user_contents)
    
    # 改为调用llm_generate函数，实现和query相同的效果
    response = llm_generate(input_user,scale_assistant_contents_v2)
                            
                            
    response = response.split("\n")

    corrected_names = []
    corrected_sizes = []
    for idx, line in enumerate(response):
        if "```yaml" in line:
            for idx2 in range(idx + 1, len(response)):
                line2 = response[idx2]
                if "```" in line2:
                    break
                line2 = line2.split(", ")
                corrected_names.append(line2[0].lower())
                sizes = line2[2:]
                if len(sizes) > 1:
                    corrected_sizes.append([float(x) for x in sizes])
                else:
                    corrected_sizes.append(float(sizes[0]))

    # replace the size in yaml
    for obj in config:
        if "type" in obj:
            if obj["type"] == "mesh" or obj["type"] == "urdf":
                obj["size"] = corrected_sizes[corrected_names.index(obj["name"].lower())]

    return config

def parse_joint_angle_response(response):
    joint_values = {}
    response = response.split("\n")
    for l_idx, line in enumerate(response):
        if line.lower().startswith("```joint values"):
            for l_idx_2 in range(l_idx + 1, len(response)):
                if response[l_idx_2].lower().startswith("```"):
                    break
                if response[l_idx_2].lower().strip() == "none":
                    continue
                joint_name, joint_value = response[l_idx_2].split(":")
                joint_values[joint_name.strip().lstrip()] = joint_value.strip().lstrip()

    return joint_values

def parse_spatial_relationship_response(response):
    spatial_relationships = []
    response = response.split("\n")
    for l_idx, line in enumerate(response):
        if line.lower().startswith("```spatial relationship"):
            for l_idx_2 in range(l_idx + 1, len(response)):
                if response[l_idx_2].lower().startswith("```"):
                    break
                if response[l_idx_2].lower().strip() == "none":
                    continue
                spatial_relationships.append(response[l_idx_2].strip().lstrip().lower())

    return spatial_relationships