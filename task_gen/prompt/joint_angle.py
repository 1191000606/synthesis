def get_joint_angle_prompt(task_attr):
    return f"""Your goal is to set the joint angles of some articulated objects to the right value in the initial state, given a task. The task is for a robot arm to learn the corresponding skills to manipulate the articulated object. 

The input to you will include the task name, a short description of the task, the articulation tree of the articulated object, a semantic file of the articulated object, and the links and joints of the articulated objects that will be involved in the task.

You should output for each joint involved in the task, what joint value it should be set to. You should output a number in the range [0, 1], where 0 corresponds to the lower limit of that joint angle, and 1 corresponds to the upper limit of the joint angle. You can also output a string of "random", which indicates to sample the joint angle within the range.

By default, the joints in an object are set to their lower joint limits. You can assume that the lower joint limit corresponds to the natural state of the articulated object. E.g., for a door's hinge joint, 0 means it is closed, and 1 means it is open. For a lever, 0 means it is unpushed, and 1 means it is pushed to the limit. 

Here are some examples:

Input:
Task name: Close the door
Description: The robot arm will close the door after it was opened. 

```door articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_1 child_link: link_0
joint_name: joint_1 joint_type: fixed parent_link: base child_link: link_1
joint_name: joint_2 joint_type: revolute parent_link: link_0 child_link: link_2
```

```door semantics
link_0 hinge rotation_door
link_1 static door_frame
link_2 hinge rotation_door
```

Links: 
- link_0: link_0 is the door. This is the part of the door assembly that the robot needs to interact with.

Joints:
- joint_0: Joint_0 is the revolute joint connecting link_0 (the door) as per the articulation tree. The robot needs to actuate this joint cautiously to ensure the door is closed.

Output:
The goal is for the robot arm to learn to close the door after it is opened. Therefore, the door needs to be initially opened, thus, we are setting its value to 1, which corresponds to the upper joint limit. 

```joint values
joint_0: 1
```


Another example:
Task name: Turn Off Faucet
Description: The robotic arm will turn the faucet off by manipulating the switch

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
For the robot to learn to turn off the faucet, it cannot be already off initially. Therefore, joint_1 should be set to its upper joint limit, or any value that is more than half of the joint range, e.g., 0.8.

```joint values
joint_1: 0.8
```


One more example:
Task name: Store an item inside the Drawer
Description: The robot arm picks up an item and places it inside the drawer of the storage furniture

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
This task involves putting one item into the drawer of the storage furniture. The robot needs to first open the drawer, put the item in, and then close it. Since the articulated object is initialized with the lower joint limit, i.e., the drawer is initially closed, it aligns with the task where the robot needs to first learn to open the drawer. Therefore, no particular joint angle needs to be set, and we just output None. 

```joint values
None
```


One more example:
Task name: Direct Lamp light
Description: The robot positions both the head and rotation bar to direct the light at a specific object or area

```Lamp articulation tree
links: 
base
link_0
link_1
link_2
link_3

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_3 child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_0 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2
joint_name: joint_3 joint_type: revolute parent_link: link_2 child_link: link_3
```

```Lamp semantics
link_0 hinge rotation_bar
link_1 hinge head
link_2 free lamp_base
link_3 hinge rotation_bar
```

Links:
- link_0: This is the rotation bar. It's necessary to direct the lamp light toward a specific area.
- link_1: This is the lamp head. It's necessary to direct the lamp light toward a specific area.

Joints:
- joint_0: This joint connects the rotation bar. By actuating this joint, the robot can direct the light.
- joint_1: This joint connects the lamp head. By actuating this joint, the robot can direct the light.

Output:
The task involves directing the lamp light at a specific area. The robot needs to learn to manipulate both the rotation bar and the lamp head to achieve this. Therefore, we need to set the initial joint angles such that the lamp is not already directed at the desired area. We can set both joint_0 and joint_1 to be randomly sampled.

```joint values
joint_0: random
joint_1: random
```

Can you do it for the following task:
Task name: {task_attr["task_name"]}
Description: {task_attr["task_description"]}

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
