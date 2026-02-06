def get_task_gen_prompt(object_category, articulation_tree, semantics):
    return f"""I will give you an articulated object, with its articulation tree and semantics. Your goal is to imagine some tasks that a robotic arm can perform with this articulated object in household scenarios. You can think of the robotic arm as a Franka Panda robot. The task will be built in a simulator for the robot to learn it. 

Focus on manipulation or interaction with the object itself. Sometimes the object will have functions, e.g., a microwave can be used to heat food, in these cases, feel free to include other objects that are needed for the task. 
Please do not think of tasks that try to assemble or disassemble the object. Do not think of tasks that aim to clean the object or check its functionality. 

For each task you imagined, please write in the following format: 
Task name: the name of the task.
Description: a basic description of the task. 
Additional objects: Additional objects other than the provided articulated object required for completing the task. 
Links: Links of the articulated object that is required to perform the task. 
- Link 1: reasons why this link is needed for the task
- Link 2: reasons why this link is needed for the task
- …
Joints: Joints of the articulated object that is required to perform the task. 
- Joint 1: reasons why this joint is needed for the task
- Joint 2: reasons why this joint is needed for the task
- …


Example Input: 
```Oven articulation tree
links: 
base
link_0
link_1
link_2
link_3
link_4
link_5
link_6
link_7

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_7 child_link: link_0
joint_name: joint_1 joint_type: continuous parent_link: link_7 child_link: link_1
joint_name: joint_2 joint_type: continuous parent_link: link_7 child_link: link_2
joint_name: joint_3 joint_type: continuous parent_link: link_7 child_link: link_3
joint_name: joint_4 joint_type: continuous parent_link: link_7 child_link: link_4
joint_name: joint_5 joint_type: continuous parent_link: link_7 child_link: link_5
joint_name: joint_6 joint_type: continuous parent_link: link_7 child_link: link_6
joint_name: joint_7 joint_type: fixed parent_link: base child_link: link_7
```

```Oven semantics
link_0 hinge door
link_1 hinge knob
link_2 hinge knob
link_3 hinge knob
link_4 hinge knob
link_5 hinge knob
link_6 hinge knob
link_7 heavy oven_body
```

Example output:

Task name: Open Oven Door
Description: The robotic arm will open the oven door.
Additional objects: None
Links:
- link_0: from the semantics, this is the door of the oven. The robot needs to approach this door in order to open it. 
Joints: 
- joint_0: from the articulation tree, this is the revolute joint that connects link_0. Therefore, the robot needs to actuate this joint for opening the door.


Task name: Adjust Oven Temperature
Description: The robotic arm will turn one of the oven's hinge knobs to set a desired temperature.
Additional objects: None
Links:
- link_1: the robot needs to approach link_1, which is assumed to be the temperature knob, to rotate it to set the temperature.
Joints:
- joint_1: joint_1 connects link_1 from the articulation tree. The robot needs to actuate it to rotate link_1 to the desired temperature.


Task name: Heat a hamburger inside Oven
Description: The robot arm places a hamburger inside the oven, and sets the oven temperature to be appropriate for heating the hamburger.
Additional objects: hamburger
Links:
- link_0: link_0 is the oven door from the semantics. The robot needs to open the door in order to put the hamburger inside the oven.
- link_1: the robot needs to approach link_1, which is the temperature knob, to rotate it to set the desired temperature.
Joints:
- joint_0: from the articulation tree, this is the revolute joint that connects link_0 (the door). Therefore, the robot needs to actuate this joint for opening the door.
- joint_1: from the articulation tree, joint_1 connects link_1, which is the temperature knob. The robot needs to actuate it to rotate link_1 to the desired temperature.


Task name: Set Oven Timer
Description: The robot arm turns a timer knob to set cooking time for the food.
Additional objects: None.
Links: 
- link_2: link_2 is assumed to be the knob for controlling the cooking time. The robot needs to approach link_2 to set the cooking time.
Joints:
- joint_2: from the articulation tree, joint_2 connects link_2. The robot needs to actuate joint_2 to rotate link_2 to the desired position, setting the oven timer.


Can you do the same for the following object:
```{object_category} articulation tree
{articulation_tree}
```

```{object_category} semantics
{semantics}
```

Do not apply bold formatting to any text; in other words, do not include ** in your answer.

Please follow the format of the sample above when answering so that I can parse your response.
"""
