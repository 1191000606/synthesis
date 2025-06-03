def get_scene_gen_prompt(task_name, task_description, object_category, additional_object, articulation_tree, semantics):
    return f"""
I need you to describe the initial scene configuration for a given task in the following format, using a yaml file. This yaml file will help build the task in a simulator. The task is for a mobile Franka panda robotic arm to learn a manipulation skill in the simulator. The Franka panda arm is mounted on a floor, at location (1, 1, 0). It can move freely on the floor. The z axis is the gravity axis. 

The format is as follows:
```yaml 
- use_table: whether the task requires using a table. This should be decided based on common sense. If a table is used, its location will be fixed at (0, 0, 0). The height of the table will be 0.6m. Usually, if the objects involved in the task are usually placed on a table (not directly on the ground), then the task requires using a table.
# for each object involved in the task, we need to specify the following fields for it.
- type: mesh
  name: name of the object, so it can be referred to in the simulator
  size: describe the scale of the object mesh using 1 number in meters. The scale should match real everyday objects. E.g., an apple is of scale 0.08m. You can think of the scale to be the longest dimension of the object. 
  lang: this should be a language description of the mesh. The language should be a concise description of the object, such that the language description can be used to search an existing database of objects to find the object.
  path: this can be a string showing the path to the mesh of the object. 
  on_table: whether the object needs to be placed on the table (if there is a table needed for the task). This should be based on common sense and the requirement of the task. E.g., a microwave is usually placed on the table.
  center: the location of the object center. If there isn't a table needed for the task or the object does not need to be on the table, this center should be expressed in the world coordinate system. If there is a table in the task and the object needs to be placed on the table, this center should be expressed in terms of the table coordinate, where (0, 0, 0) is the lower corner of the table, and (1, 1, 1) is the higher corner of the table. In either case, you should try to specify a location such that there is no collision between objects.
  movable: if the object is movable or not in the simulator due to robot actions. This option should be false for most tasks; it should be true only if the task specifically requires the robot to move the object. This value can also be missing, which means the object is not movable.
```

An example input includes the task names, task descriptions, and objects involved in the task. I will also provide with you the articulation tree and semantics of the articulated object. 
This can be useful for knowing what parts are already in the articulated object, and thus you do not need to repeat those parts as separate objects in the yaml file.

Your task includes two parts:
1. Output the yaml configuration of the task.
2. Sometimes, the task description / objects involved will refer to generic/placeholder objects, e.g., to place an "item" into the drawer, and to heat "food" in the microwave. In the generated yaml config, you should change these placeholder objects to be concrete objects in the lang field, e.g., change "item" to be a toy or a pencil, and "food" to be a hamburger, a bowl of soup, etc. 

Example input:
Task Name: Insert Bread Slice 
Description: The robotic arm will insert a bread slice into the toaster.
Objects involved: Toaster, bread slice. Only the objects specified here should be included in the yaml file.

```Toaster articulation tree
links: 
base
link_0
link_1
link_2
link_3
link_4
link_5

joints: 
joint_name: joint_0 joint_type: continuous parent_link: link_5 child_link: link_0
joint_name: joint_1 joint_type: prismatic parent_link: link_5 child_link: link_1
joint_name: joint_2 joint_type: prismatic parent_link: link_5 child_link: link_2
joint_name: joint_3 joint_type: prismatic parent_link: link_5 child_link: link_3
joint_name: joint_4 joint_type: prismatic parent_link: link_5 child_link: link_4
joint_name: joint_5 joint_type: fixed parent_link: base child_link: link_5
```

```Toaster semantics
link_0 hinge knob
link_1 slider slider
link_2 slider button
link_3 slider button
link_4 slider button
link_5 free toaster_body
```


An example output:
```yaml
- use_table: True ### Toaster and bread are usually put on a table. 
- type: mesh
  name: "Toaster"
  on_table: True # Toasters are usually put on a table.
  center: (0.1, 0.1, 0) # Remember that when an object is placed on the table, the center is expressed in the table coordinate, where (0, 0, 0) is the lower corner and (1, 1, 1) is the higher corner of the table. Here we put the toaster near the lower corner of the table.  
  size: 0.35 # the size of a toaster is roughly 0.35m
  lang: "a common toaster"
  path: "toaster.urdf"
- type: mesh
  name: "bread slice"
  on_table: True # Bread is usually placed on the table as well. 
  center: (0.8, 0.7, 0) # Remember that when an object is placed on the table, the center is expressed in the table coordinate, where (0, 0, 0) is the lower corner and (1, 1, 1) is the higher corner of the table. Here we put the bread slice near the higher corner of the table.  
  size: 0.1 # common size of a bread slice 
  lang: "a slice of bread"
  Path: "bread_slice.obj"
```

Another example input:
Task Name: Removing Lid From Pot
Description: The robotic arm will remove the lid from the pot.
Objects involved: KitchenPot. Only the objects specified here should be included in the yaml file.

```KitchenPot articulation tree
links: 
base
link_0
link_1

joints: 
joint_name: joint_0 joint_type: prismatic parent_link: link_1 child_link: link_0
joint_name: joint_1 joint_type: fixed parent_link: base child_link: link_1
```

```KitchenPot semantics
link_0 slider lid
link_1 free pot_body
```
Output:
```yaml
- use_table: True # A kitchen pot is usually placed on the table.
- type: mesh
  name: "KitchenPot"
  on_table: True # kitchen pots are usually placed on a table. 
  center: (0.3, 0.6, 0) # Remember that when an object is placed on the table, the center is expressed in the table coordinate, where (0, 0, 0) is the lower corner and (1, 1, 1) is the higher corner of the table. Here we put the kitchen pot just at a random location on the table.  
  size: 0.28 # the size of a common kitchen pot is roughly 0.28m
  lang: "a common kitchen pot"
  path: "kitchen_pot.urdf"
```
Note in this example, the kitchen pot already has a lid from the semantics file. Therefore, you do not need to include a separate lid in the yaml file.


One more example input:
Task Name: Push the chair.
Description: The robotic arm will push and move the chair to a target location.
Objects involved: A chair. Only the objects here should be included in the yaml file.

```Chair articulation tree
links: 
base
link_0
link_1

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_1 child_link: link_0
joint_name: joint_1 joint_type: fixed parent_link: base child_link: link_1
```

```Chair semantics
link_0 hinge seat
link_1 free leg
```

Output:
```yaml
- use_table: False # A chair is usually just on the ground
- type: mesh
  name: "Chair"
  on_table: False # An oven is usually just placed on the floor.
  center: (1.0, 0, 0) # Remember that when not on a table, the center is expressed in the world coordinate. Since the robot is at (1, 1, 0) and the table is at (0, 0, 0), we place the oven at (1.8, 2, 0) to avoid collision with the table and the robot.
  size: 1.2 # the size of an oven is roughly 0.9m
  lang: "a standard chair"
  path: "chair.urdf"
  movable: True # here the task requires the robot to push the chair, so the chair has to be moveable.
```
Note in the above example we set the chair to be moveable so the robot can push it for executing the task.

Another example:
Task Name: Put an item into the box drawer
Description: The robot will open the drawer of the box, and put an item into it.
Objects involved: A box with drawer, an item to be placed in the drawer. 

```Box articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_2 child_link: link_0
joint_name: joint_1 joint_type: prismatic parent_link: link_2 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2
```

```Box semantics
link_0 hinge rotation_lid
link_1 slider drawer
link_2 free box_body
```

Output:
```yaml
-   use_table: true
-   center: (0.5, 0.5, 0)
    lang: "a wooden box"
    name: "Box"
    on_table: true
    path: "box.urdf"
    size: 0.3
    type: urdf
-   path: "item.obj"
    center: (0.2, 0.4, 0)
    lang: "A toy" # Note here, we changed the generic/placeholder "item" object to be a more concrete object: a toy. 
    name: "Item"
    on_table: true
    size: 0.05
    type: mesh
```

One more example:
Task Name: Fetch item from refrigerator
Description: The robot will open the refrigerator door, and fetch an item from the refrigerator.
Objects involved: A refrigerator, an item to be fetched from the refrigerator.

```Refrigerator articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: fixed parent_link: base child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_0 child_link: link_1
joint_name: joint_2 joint_type: revolute parent_link: link_0 child_link: link_2
```

```Refrigerator semantics
link_0 heavy refrigerator_body
link_1 hinge door
link_2 hinge door
```

Output:
```yaml
-   use_table: true # the fetched item should be placed on the table, after it's moved out of the refrigerator.
-   center: (1.0, 0.2, 0) # Remember that when not on a table, the center is expressed in the world coordinate. Since the robot is at (1, 1, 0) and the table is at (0, 0, 0), we place the oven at (1.8, 2, 0) to avoid collision with the table and the robot.
    lang: a common two-door refrigerator
    name: Refrigerator
    on_table: false # the refrigerator is usually placed on the floor.
    path: refrigerator.urdf
    reward_asset_path: '10612'
    size: 1.8
    type: urdf
-   center: (1.0, 0.2, 0.5) # the soda can is initially placed inside the refrigerator.
    lang: a can of soda
    name: Item
    on_table: false # the item is initially placed inside the refrigerator
    path: soda_can.obj
    size: 0.2
    type: mesh
```

Rules: 
- You do not need to include the robot in the yaml file.
- The yaml file should only include the objects listed in "Objects involved".
- Sometimes, the task description / objects involved will refer to generic/placeholder objects, e.g., to place an "item" into the drawer, and to heat "food" in the microwave. In the generated yaml config, you should change these placeholder objects to be concrete objects in the lang field, e.g., change "item" to be a toy or a pencil, and "food" to be a hamburger, a bowl of soup, etc. 


Can you do this for the following task:
Task Name: {task_name}
Description: {task_description}
Objects involved: {object_category}, {additional_object}

```{object_category} articulation tree
{articulation_tree}
```

```{object_category} semantics
{semantics}
```
"""
