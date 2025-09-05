# dataset/下载地址
原始的PartNet-Mobility Dataset数据集下载地址：https://sapien.ucsd.edu/downloads（已下载解压至./data/partnet/partnet-mobility-v0/dataset）

需要注册账号，然后申请下载权限，审核通过后即可下载。

由RoboGen处理后的数据集下载地址：https://drive.google.com/file/d/1d-1txzcg_ke17NkHKAolXlfDnmPePFc6/view（已下载解压至./data/partnet/dataset）

TODO：RoboGen做了哪些处理？

# category.txt下载地址
这个文件应该是category_to_ids.json提取的，这里有46类

可以从这里看到全部的类别 https://sapien.ucsd.edu/browse

然而PartNet的v0版本只有24个类别

# category_to_ids.json下载地址
这个文件是从RoboGen下载的（https://github.com/Genesis-Embodied-AI/RoboGen/blob/main/data/partnet_mobility_dict.json）

根据https://sapien.ucsd.edu/browse，例如bottle类别中包括id为3380的，这个在./data/partnet/dataset确实存在，确实是bottle，但是category_to_ids.json中没有3380这个id。

检查一下这个是为什么，https://partnet.cs.stanford.edu/这里面什么也看不出来