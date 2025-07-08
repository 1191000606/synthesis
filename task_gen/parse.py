import yaml

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

            involved_links = []
            joint_start_line = 0
            for link_idx in range(l_idx + 4, len(task_response)):
                if task_response[link_idx].lower().startswith("joints:"):
                    joint_start_line = link_idx
                    break
                else:
                    involved_links.append(task_response[link_idx][2:].strip())
            links.append("\n".join(involved_links))

            involved_joints = []
            for joint_idx in range(joint_start_line + 1, len(task_response)):
                if not task_response[joint_idx].lower().startswith("- "):
                    break
                else:
                    involved_joints.append(task_response[joint_idx][2:].strip())
            joints.append("\n".join(involved_joints))

    return task_names, task_descriptions, additional_objects, links, joints

def parse_response_to_get_yaml(response):
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
    return yaml.safe_load(yaml_string)


def parse_scale(response):
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
                corrected_sizes.append(line2[1].strip())

    return corrected_names, corrected_sizes

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

    return {"spatial_relationships": spatial_relationships}
