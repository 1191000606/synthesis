def get_distractor_prompt(task_name, task_description, scene_config):
    scene_config_str = get_scene_config_str(scene_config)
    
    return f"""Given a task, which is for a mobile Franka panda robotic arm to learn a manipulation skill in the simulator, your goal is to add more objects into the task scene such that the scene looks more realistic. The Franka panda arm is mounted on a floor, at location (1, 1, 0). It can move freely on the floor. The z axis is the gravity axis. 

The input to you includes the following:
Task name, task description, the essential objects involved in the task, and a config describing the current task scene, which contains only the essential objects needed for the task. The config is a yaml file in the following format:

```yaml 
- use_table: whether the task requires using a table. This should be decided based on common sense. If a table is used, its location will be fixed at (0, 0, 0). The height of the table will be 0.6m.
# for each object involved in the task, we need to specify the following fields for it.
- type: mesh or urdf. If the object is articulated, then it should be urdf; otherwise, it should be mesh.
  name: name of the object, so it can be referred to in the simulator.
  on_table: whether the object needs to be placed on the table (if there is a table needed for the task). This should be based on common sense and the requirement of the task.
  size: describe the scale of the object mesh using 1 number in meters. The scale should match real everyday objects. E.g., an apple is of scale 0.08m. You can think of the scale to be the longest dimension of the object.
  center: the location of the object center. If there isn't a table needed for the task or the object does not need to be on the table, this center should be expressed in the world coordinate system. If there is a table in the task and the object needs to be placed on the table, this center should be expressed in terms of the table coordinate, where (0, 0, 0) is the lower corner of the table, and (1, 1, 1) is the higher corner of the table. In either case, you should try to specify a location such that there is no collision between objects.
  lang: this should be a language description of the object. The language should be a bit detailed, such that the language description can be used to search an existing database of objects to find the object.
  path: this can be a string showing the path to the asset of the object.
  movable: if the object is movable or not in the simulator due to robot actions. This option should be false for most tasks; it should be true only if the task specifically requires the robot to move the object.
```

Your task is to think about what other distractor objects can be added into the scene to make the scene more complex and realistic for the robot to learn the task. These distractor objects are not necessary for the task itself, but their existence makes the scene look more interesting and complex. You should output the distractor objects using the same format as the input yaml file. You should try to put these distractor objects at locations such that they don’t collide with objects already in the scene. 

Here is one example:

Input:

Task name: Heat up a bowl of soup in the microwave
Task description: The robot will grab the soup and move it into the microwave, and then set the temperature to heat it.
Objects involved: Microwave, a bowl of soup
Config:
```yaml
- use_table: true
- type: urdf
  name: Microwave
  on_table: true
  size: 0.6
  center: (0.3, 0.7, 0)
  lang: A standard microwave with a turntable and digital timer
  path: microwave.urdf
  movable: false
- type: mesh
  name: Bowl of Soup
  on_table: true
  size: 0.15
  center: (0.2, 0.2, 0)
  lang: A ceramic bowl full of soup
  path: bowl_soup.obj
  moveable: true
```

Output: 
```yaml
- type: mesh
  name: plate # a plate is a common object placed when there is microwave and bowl of soup, in a kitchen setup
  on_table: True
  size: 0.15 # a plate is usually of scale 0.15m
  center: (0.8, 0.8, 0)
  lang: a common kitchen plate
  path: "plate.obj"
  movable: False # The distractor objects are normally not be movable in the simulator.
- type: mesh
  name: spoon # a spoon is also a common object in a kitchen setup
  on_table: True
  size: 0.1 # a spoon is usually of scale 0.1m
  center: (0.5, 0.2, 0)
  lang: a common sponge
  path: "sponge.obj"
  movable: False # The distractor objects are normally not be movable in the simulator.
- type: mesh
  name: Oven
  on_table: False # an oven is usually a standalone object on the floor
  size: 0.8 # an oven is usually of scale 0.8m
  center: (1.8, 0.5, 0) # remember robot is at (1, 1, 0) and table is at (0, 0, 0). So the oven is placed at (1.8, 0.5, 0) in the world coordinate system to avoid collision with other objects.
  lang: a kitchen oven
  path: "oven.obj"
  moveable: False
```

Can you do it for the following task:
Task name: {task_name}
Task description: {task_description}
Config:
```yaml
{scene_config_str}
```
"""

def get_scene_config_str(scene_config):
    config_lines = []
    for obj in scene_config:
        if 'use_table' in obj:
            config_lines.append(f"- use_table: {obj['use_table']}")
            continue

    for obj in scene_config:
        if "type" in obj and "name" in obj:
            config_lines.append(f"- type: {obj['type']}")
            config_lines.append(f"  name: {obj['name']}")
            config_lines.append(f"  on_table: {obj['on_table']}")
            config_lines.append(f"  size: {obj['size']}")
            config_lines.append(f"  center: {obj['center']}")
            config_lines.append(f"  lang: {obj['lang']}")
            config_lines.append(f"  path: {obj['path']}")
            config_lines.append(f"  movable: {obj['movable']}")
    return "\n".join(config_lines)