def get_scale_prompt(configs):
    task_description = ""
    object_scale_str = ""

    for config in configs:
        if "task_description" in config:
            task_description = config["task_description"]
            continue

        if "name" in config and "size" in config:
            object_scale_str += f"{config['name'].lower()}, {str(config['size'])}\n"

    object_scale_str = object_scale_str[:-1]

    return f"""A robotic arm is trying to manipulate some objects to learn corresponding skills in a simulator. However, the size of the objects might be wrong. Your task is to adjust the size of the objects, such that they match each other when interact with each other; and the size should also match what is commonly seen in everyday life, in household scenarios. 

Now I will give you the name of the task, the object and their sizes, please correct any unreasonable sizes. 

Objects are represented using a mesh file, you can think of size as the longest dimension of the object. 

I will write in the following format:
```
Task: task description
obj1, size 
obj2, size
```

Please reply in the following format:
explanations of why some size is not reasonable.

```yaml
obj1, corrected_size
obj2, corrected_radius
```

Here is an example:
Input: 
```
Task: The robotic arm lowers the toilet seat from an up position to a down position
Toilet, 0.2
```

Output:
A toilet is usually 0.6 - 0.8m in its back height, so the size is not reasonable -- it is a bit too small. Below is the corrected size.

```yaml
Toilet, 0.7
```

Another example:
Input:
```
Task: Fill a cup with water under the faucet
Faucet, 0.25
Cup, 0.3
```

Output:
The size of the faucet makes sense. However, the size of the cup is too large for 2 reasons: it does not match the size of tha faucet for getting water under the faucet; and it is not a common size of cup in everyday life. Below is the corrected size.

```yaml
Faucet, 0.25 
Cup, 0.12 
```

One more example to show that even if no change is needed, you should still reply with the same size.
Input:
```
Task: Open Table Drawer The robotic arm will open a table drawer
table, 0.8
```

Output:
The size of the table is reasonable, so no change is needed.

```yaml
table, 0.8
```

This is also a good example to show that sometimes, the task description might include two objects, e.g., a table and a drawer, yet there is only one object size provided (here the table). You just need to adjust the sizes of the provided objects, instead of asking why other objects are not includes.

Another example input:
```
Task: Heat up a bowl of soup in the microwave
plate, 0.3
sponge, 0.1
oven, 0.4
```

Output:
The size of the sponge make sense. However, the size of the plate is too big, and the size of the oven is too small.

```yaml
plate, 0.15
sponge, 0.1
oven, 0.8
```

Can you do it for the following task:
```
Task: {task_description}
{object_scale_str}
```
"""
