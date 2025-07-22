import os
import yaml
import os.path as osp
import json
import multiprocessing
import objaverse
import trimesh
import numpy as np
import pybullet as p
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests
from bardapi import Bard
from llm import llm_generate
from pathlib import Path
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = "Salesforce/blip2-flan-t5-xl"     # 也可用 blip2-opt-2.7b 或 blip2-flan-t5-xxl
processor = Blip2Processor.from_pretrained(ckpt)
blip2 = Blip2ForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
blip2.to(device).eval()
sentence_bert_model = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"
root_path = '/home/szwang/synthesis'
objaverse_csv = pd.read_csv(osp.join(root_path, 'data/objaverse/Cap3D_automated_Objaverse.csv'))
objaverse_csv = objaverse_csv.dropna()
objaverse_csv_uids = list(objaverse_csv.iloc[:, 0].values)
objaverse_csv_annotations = list(objaverse_csv.iloc[:, 1].values)
objaverse_csv_annotations_embeddings = torch.load((osp.join(root_path, 'data/objaverse/data/cap3d_sentence_bert_embeddings.pt')), weights_only=False)
tag_uids = []
tag_embeddings = []
tag_descriptions = [] 
num_chunks = 31
for idx in range(num_chunks):
    uids = torch.load(osp.join(root_path,"data/objaverse/data/default_tag_uids_{}.pt".format(idx)),weights_only=False,map_location="cpu")#.to(device)
    embeddings = torch.load(osp.join(root_path,"data/objaverse/data/default_tag_embeddings_{}.pt".format(idx)),weights_only=False,map_location="cpu")#.to(device)
    descriptions = torch.load(osp.join(root_path,"data/objaverse/data/default_tag_names_{}.pt".format(idx)),weights_only=False,map_location="cpu")#.to(device)
    tag_uids = tag_uids + uids
    tag_descriptions = tag_descriptions + descriptions
    tag_embeddings.append(embeddings)

def check_text_similarity(text, check_list=None, check_embeddings=None):
    global sentence_bert_model
    if sentence_bert_model is None:
        sentence_bert_model = SentenceTransformer('all-mpnet-base-v2')

    #Sentences are encoded by calling model.encode()
    with torch.no_grad():
        emb1 = sentence_bert_model.encode(text)
        if check_embeddings is None:
            emb_to_check = sentence_bert_model.encode(check_list)
        else:
            emb_to_check = check_embeddings
        cos_sim = util.cos_sim(emb1, emb_to_check)

    return cos_sim.cpu().numpy()

def find_uid(obj_descrption, candidate_num=10, debug=False, task_name=None, task_description=None):
    uids = text_to_uid_dict.get(obj_descrption, None)
    all_uid_candidates = text_to_uid_dict.get(obj_descrption + "_all", None)

    if uids is None:
        print("searching whole objaverse for: ", obj_descrption)
        similarities = []
        for idx in range(num_chunks):
            similarity = check_text_similarity(obj_descrption, None, check_embeddings=tag_embeddings[idx])
            similarity = similarity.flatten()
            similarities.append(similarity)
        
        similarity = check_text_similarity(obj_descrption, None, check_embeddings=objaverse_csv_annotations_embeddings)
        similarity = similarity.flatten()
        similarities.append(similarity)
        similarities = np.concatenate(similarities)

        all_uids = tag_uids + objaverse_csv_uids
        all_description = tag_descriptions + objaverse_csv_annotations

        sorted_idx = np.argsort(similarities)[::-1]

        usable_uids = []
        all_uid_candidates = [all_uids[sorted_idx[i]] for i in range(candidate_num)]
        for candidate_idx in range(candidate_num):
            print("{} candidate {} similarity: {} {}".format("=" * 10, candidate_idx, similarities[sorted_idx[candidate_idx]], "=" * 10))
            print("found uid: ", all_uids[sorted_idx[candidate_idx]])
            print("found description: ", all_description[sorted_idx[candidate_idx]])

            candidate_uid = all_uids[sorted_idx[candidate_idx]]
            bard_verify_result = verify_objaverse_object(obj_descrption, candidate_uid, task_name=task_name, task_description=task_description) # TO DO: add support for including task name in the checking process
            print("{} Bard thinks this object is usable: {} {}".format("=" * 20, bard_verify_result, "=" * 20))
            if bard_verify_result:
                usable_uids.append(candidate_uid)
            # usable_uids.append(candidate_uid)

        if len(usable_uids) == 0:
            print("no usable objects found for {} skipping this task!!!".format(obj_descrption))
            usable_uids = all_uid_candidates

        text_to_uid_dict[obj_descrption] = usable_uids
        text_to_uid_dict[obj_descrption + "_all"] = all_uid_candidates
        with open(osp.join(root_path, 'data/objaverse/text_to_uid.json'), 'w') as f:
            json.dump(text_to_uid_dict, f, indent=4)
        return usable_uids
    else:
        return uids

def verify_objaverse_object(object_name, uid, task_name=None, task_description=None, use_bard=False, use_blip2=True):
    annotations = objaverse.load_annotations([uid])[uid]
    thumbnail_urls = annotations['thumbnails']["images"]

    max_size = -1000
    max_url = -1
    for dict in thumbnail_urls:
        width = dict["width"]
        if width > max_size:
            max_size = width
            max_url = dict["url"]
    if max_url == -1: # TO DO: in this case, we should render the object using blender to get the image.
        return False
    
    # download the image from the url
    try: 
        raw_image = Image.open(requests.get(max_url, stream=True).raw).convert('RGB')
    except:
        return False
    
    if not os.path.exists(osp.join(root_path,'data/objaverse/data/images')):
        os.makedirs(osp.join(root_path,'data/objaverse/data/images'))
        
    raw_image.save(osp.join(root_path,"data/objaverse/data/images/{}.jpeg".format(uid)))
    # bard_image = open("objaverse_utils/data/images/{}.jpeg".format(uid), "rb").read()
    bard_image = open(osp.join(root_path,"data/objaverse/data/images/{}.jpeg".format(uid)), "rb").read()

    descriptions = []
    if use_bard:
        bard_description = bard_verify(bard_image)
        descriptions.append(bard_description)
    if use_blip2:
        blip2_description = blip2_caption(raw_image)
        descriptions.append(blip2_description)

    gpt_results = []

    for description in descriptions:
        if description:
            query_string = """
            A robotic arm is trying to solve a task to learn a manipulation skill in a simulator.
        We are trying to find the best objects to load into the simulator to build this task for the robot to learn the skill.
        The task the robot is trying to learn is: {}. 
        A more detailed description of the task is: {}.
        As noted, to build the task in the simulator, we need to find this object: {}.
        We are retrieving the object from an existing database, which provides some language annotations for the object.
        With the given lanugage annotation, please think if the object can be used in the simulator as {} for learning the task {}.

        This is the language annotation:
        {}

        Please reply first with your reasoning, and then a single line with "**yes**" or "**no**" to indicate whether this object can be used.
        """.format(task_name, task_description, object_name, object_name, task_name, description)
        
            if not os.path.exists(osp.join(root_path,'data/objaverse/data/debug')):
                os.makedirs(osp.join(root_path,'data/objaverse/data/debug'))
                

            res = llm_generate(query_string)
            
            responses = res.split("\n")

            useable = False
            for l_idx, line in enumerate(responses):
                if "yes" in line.lower():
                    useable = True
                    break

            gpt_results.append(useable)

    return np.alltrue(gpt_results)


def bard_verify(image):
    token = "" # replace with your token
    session = requests.Session()
    session.headers = {
                "Host": "bard.google.com",
                "X-Same-Domain": "1",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "Origin": "https://bard.google.com",
                "Referer": "https://bard.google.com/",
            }
    session.cookies.set("__Secure-1PSID", token) 
    bard = Bard(token=token, session=session)

    
    query_string = """I will show you an image. Please describe the content of the image. """

    print("===================== querying bard: ==========================")
    print(query_string)
    res = bard.ask_about_image(query_string, image)
    description = res['content']
    print("bard description: ", description)
    print("===============")
    return description
    
    
'''
def blip2_caption(image):
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    # generate caption
    res = model.generate({"image": image})
    return res[0]
'''

### 这里进行了修改，原来的代码使用了from lavis.models import load_model_and_preprocess，这里我直接下载了blip模型的权重来实现一样的功能
### 或许它原来的api调用会更快吗？如果是的话，既然当前环境是在不兼容，可以再额外为lavis.models单独包装一个接口


def blip2_caption(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip2.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True).strip()

if osp.exists(osp.join(root_path, 'data/objaverse/text_to_uid.json')):
    with open(osp.join(root_path, 'data/objaverse/text_to_uid.json'), 'r') as f:
        text_to_uid_dict = json.load(f)
else:
    text_to_uid_dict = {}


def down(folder_path):
    # folder_path = os.path.join("..", "generated_hpc", folder_path)
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("路径无效或不是一个有效的文件夹。")
        return
    
    # 遍历文件夹下所有文件
    for filename in os.listdir(folder_path):
        # 只处理yaml文件
        if filename.endswith('.yaml'):
            file_path = os.path.join(folder_path, filename)
            # 调用下载和解析函数
            download_and_parse_objavarse_obj_from_yaml_config(file_path)
    print("下载完成。")
    
    
    
def download_and_parse_objavarse_obj_from_yaml_config(config_path, candidate_num=10, vhacd=True):

    config = None
    while config is None:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

    task_name = None
    task_description = None
    for obj in config:
        if 'task_name' in obj.keys():
            task_name = obj['task_name']
            task_description = obj['task_description']
            break

    for obj in config:
        if 'type' in obj.keys() and obj['type'] == 'mesh' and 'uid' not in obj.keys():
            print("{} trying to download object: {} {}".format("=" * 20, obj['lang'], "=" * 20))
            success = down_load_single_object(obj["lang"], candidate_num=candidate_num, vhacd=vhacd, 
                                              task_name=task_name, task_description=task_description)
            if not success:
                print("failed to find suitable object to download {} quit building this task".format(obj["lang"]))
                return False
            obj['uid'] = text_to_uid_dict[obj["lang"]]
            obj['all_uid'] = text_to_uid_dict[obj["lang"] + "_all"]

            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=4)

    return True



def down_load_single_object(name, uids=None, candidate_num=5, vhacd=True, debug=False, task_name=None, task_description=None):
    if uids is None:
        if name in text_to_uid_dict:
            uids = text_to_uid_dict[name]
        else:
            print("cannot find uid for object: ", name)
            print("trying to find suitable object for: ", name)
            uids = find_uid(name, candidate_num=candidate_num, debug=debug, task_name=task_name, task_description=task_description)
            if uids is None:
                return False
    print("uids are: ", uids)

    processes = multiprocessing.cpu_count()
   
    for uid in uids:
        ###################################################################
        #save_path = osp.join("objaverse_utils/data/obj", "{}".format(uid))
        save_path = osp.join(root_path,"data/objaverse/data/obj", "{}".format(uid))
        print("save_path is: ", save_path)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        if osp.exists(save_path + "/material.urdf"):
            continue

        objects = objaverse.load_objects(
            uids=[uid],
            download_processes=processes
        )
        
        test_obj = (objects[uid])
        scene = trimesh.load(test_obj)

        try:
            trimesh.exchange.export.export_mesh(
                scene, osp.join(save_path, "material.obj")
            )
        except:
            print("cannot export obj for uid: ", uid)
            uids.remove(uid)
            if uid in text_to_uid_dict[name]:
                text_to_uid_dict[name].remove(uid)
            continue

        # we need to further parse the obj to normalize the size to be within -1, 1
        if not osp.exists(osp.join(save_path, "material_normalized.obj")):
            normalize_obj(osp.join(save_path, "material.obj"))

        # we also need to parse the obj to vhacd
        if vhacd:
            if not osp.exists(osp.join(save_path, "material_normalized_vhacd.obj")):
                run_vhacd(save_path)

        # for pybullet, we have to additionally parse it to urdf
        obj_to_urdf(save_path, scale=1, vhacd=vhacd) 

    return True

def obj_to_urdf(obj_file_path, scale=1, vhacd=True, normalized=True, obj_name='material'):
    header = """<?xml version="1.0" ?>
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
"""

    all_files = os.listdir(obj_file_path)
    png_file = None
    for x in all_files:
        if x.endswith(".png"):
            png_file = x
            break

    if png_file is not None:
        material = """
         <material name="texture">
        <texture filename="{}"/>
      </material>""".format(osp.join(obj_file_path, png_file))        
    else:
        material = """
        <material name="yellow">
            <color rgba="1 1 0.4 1"/>
        </material>
        """

    obj_file = "{}.obj".format(obj_name) if not normalized else "{}_normalized.obj".format(obj_name)
    visual = """
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
      {}
    </visual>
    """.format(osp.join(obj_file_path, obj_file), scale, scale, scale, material)

    if normalized:
        collision_file = '{}_normalized_vhacd.obj'.format(obj_name) if vhacd else "{}_normalized.obj".format(obj_name)
    else:
        collision_file = '{}_vhacd.obj'.format(obj_name) if vhacd else "{}.obj".format(obj_name)
    collision = """
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
             <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
    </collision>
  </link>
  </robot>
  """.format(osp.join(obj_file_path, collision_file), scale, scale, scale)
    


    urdf =  "".join([header, visual, collision])
    with open(osp.join(obj_file_path, "{}.urdf".format(obj_name)), 'w') as f:
        f.write(urdf)



def normalize_obj(obj_file_path):
    vertices = []
    with open(osp.join(obj_file_path), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:]])
    
    vertices = np.array(vertices).reshape(-1, 3)
    vertices = vertices - np.mean(vertices, axis=0) # center to zero
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1)) # normalize to -1, 1

    with open(osp.join(obj_file_path.replace(".obj", "_normalized.obj")), 'w') as f:
        vertex_idx = 0
        for line in lines:
            if line.startswith("v "):
                line = "v " + " ".join([str(x) for x in vertices[vertex_idx]]) + "\n"
                vertex_idx += 1
            f.write(line)
            
def run_vhacd(input_obj_file_path, normalized=True, obj_name="material"):
    p.connect(p.DIRECT)
    if normalized:
        name_in = os.path.join(input_obj_file_path, "{}_normalized.obj".format(obj_name))
        name_out = os.path.join(input_obj_file_path, "{}_normalized_vhacd.obj".format(obj_name))
        name_log = os.path.join(input_obj_file_path, "log.txt")
    else:
        name_in = os.path.join(input_obj_file_path, "{}.obj".format(obj_name))
        name_out = os.path.join(input_obj_file_path, "{}_vhacd.obj".format(obj_name))
        name_log = os.path.join(input_obj_file_path, "log.txt")
    p.vhacd(name_in, name_out, name_log)
    
    
    
    
if __name__ == "__main__":

    down("/home/szwang/synthesis/data/generated_tasks_release/Fan_101369_2025-07-09-16-22-20")