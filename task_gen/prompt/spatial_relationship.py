def get_spatial_relationship_prompt(task_attr):
    involved_objects = task_attr["object_category"]
    if task_attr["additional_object"] != "None":
        involved_objects += ", " + task_attr["additional_object"]

    return f"""Your goal is to output any special spatial relationships certain objects should have in the initial state, given a task. The task is for a robot arm to learn the corresponding skills in household scenarios.  

The input to you will include the task name, a short description of the task, objects involved in the task. If there is an articulated object involved in the task, the articulation tree of the articulated object, the semantic file of the articulated object, and the links and joints of the articulated objects that will be involved in the task. 

We have the following spatial relationships:
on, obj_A, obj_B: object A is on top of object B, e.g., a fork on the table.
in, obj_A, obj_B: object A is inside object B, e.g., a gold ring in the safe.
in, obj_A, obj_B, link_name: object A is inside the link with link_name of object B. For example, a table might have two drawers, represented with link_0, and link_1, and in(pen, table, link_0) would be that a pen is inside one of the drawers that corresponds to link_0. 

Given the input to you, you should output any needed spatial relationships of the involved objects. 

Here are some examples:

Input:
Task name: Fetch Item from Refrigerator 
Description: The robotic arm will open a refrigerator door and reach inside to grab an item and then close the door.
Objects involved: refrigerator, item

```refrigerator articulation tree
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

```refrigerator semantics
link_0 heavy refrigerator_body
link_1 hinge door
link_2 hinge door
```

Links:
- link_1: The robot needs to approach and open this link, which represents one of the refrigerator doors, to reach for the item inside.

Joints:
- joint_1: This joint connects link_1, representing one of the doors. The robot needs to actuate this joint to open the door, reach for the item, and close the door. 

Output:
The goal is for the robot arm to learn to retrieve an item from the refrigerator. Therefore, the item needs to be initially inside the refrigerator. From the refrigerator semantics we know that link_0 is the body of the refrigerator, therefore we should have a spatial relationship as the following:

```spatial relationship
In, item, refrigerator, link_0
```


Another example:
Task name: Turn Off Faucet
Description: The robotic arm will turn the faucet off by manipulating the switch
Objects involved: faucet

```Faucet articulation tree
links: 
base
link_0
link_1

joints: 
joint_name: joint_0 joint_type: fixed parent_link: base child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_0 child_link: link_1
```

```Faucet semantics
link_0 static faucet_base
link_1 hinge switch
```

Links:
- link_1: link_1 is the switch of the faucet. The robot needs to interact with this part to turn the faucet off.

Joints:
- joint_1: Joint_1 is the revolute joint connecting link_1 (the switch) as per the articulation tree. The robot needs to actuate this joint to turn the faucet off.

Output:
There is only 1 object involved in the task, thus no special spatial relationships are required.

```spatial relationship
None
```


One more example:
Task name: Store an item inside the Drawer
Description: The robot arm picks up an item and places it inside the drawer of the storage furniture.
Objects involved: storage furniture, item

```StorageFurniture articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_1 child_link: link_0
joint_name: joint_1 joint_type: fixed parent_link: base child_link: link_1
joint_name: joint_2 joint_type: prismatic parent_link: link_1 child_link: link_2
```

```StorageFurniture semantics
link_0 hinge rotation_door
link_1 heavy furniture_body
link_2 slider drawer
```

Links:
- link_2: link_2 is the drawer link from the semantics. The robot needs to open this drawer to place the item inside. 

Joints:
- joint_2: joint_2, from the articulation tree, connects to link_2 (the drawer). Thus, the robot would need to actuate this joint to open the drawer to store the item.

Output:
This task involves putting one item into the drawer of the storage furniture. The item should initially be outside of the drawer, such that the robot can learn to put it into the drawer. Therefore, no special relationships of in or on are needed. Therefore, no special spatial relationships are needed.

```spatial relationship
None
```

Can you do it for the following task: 
Task name: {task_attr["task_name"]}
Description: {task_attr["task_description"]}
Objects involved: {involved_objects}

```{task_attr["object_category"]} articulation tree
{task_attr["articulation_tree"]}
```

```{task_attr["object_category"]} semantics
{task_attr["semantics"]}
```

Links:
{task_attr["links"]}

Joints:
{task_attr["joints"]}
"""
