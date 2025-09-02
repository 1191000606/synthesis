# objaverse_xl.csv文件来源

```python
import objaverse.xl as oxl
annotations = oxl.get_annotations()
annotations.to_csv("objaverse_xl.csv", index=False)
```

这样“多此一举”的原因是，直接执行oxl.get_annotations()可能在有的电脑上会出现段错误，这个函数的返回结果是一个pandas的DataFrame对象，直接保存为csv文件后，就可以避免这个问题。

# cap3d_full_bgem3_embeddings.pkl文件来源

从https://huggingface.co/datasets/tiange/Cap3D/blob/main/Cap3D_automated_Objaverse_full.csv下载Cap3D_automated_Objaverse_full.csv，包括1.5M个3D模型及其文本描述。

```python
import pickle
import pandas

from FlagEmbedding import BGEM3FlagModel

data = pandas.read_csv("Cap3D_automated_Objaverse_full.csv", header=None, names=["uid", "caption"])

uid_list = data["uid"].tolist()
caption_list = data["caption"].tolist()

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

embeddings = model.encode(caption_list, batch_size=64, max_length=512)["dense_vecs"]

pickle.dump((uid_list, embeddings), open("cap3d_bgem3_embeddings.pkl", "wb"))
```

注意Cap3D_automated_Objaverse.csv文件（下载地址https://github.com/Genesis-Embodied-AI/RoboGen/blob/main/objaverse_utils/Cap3D_automated_Objaverse.csv 或者 https://huggingface.co/datasets/tiange/Cap3D/blob/main/misc/Cap3D_automated_Objaverse.csv）是来自论文Scalable 3D Captioning with Pretrained Models

在Scalable 3D Captioning with Pretrained Models论文之后，作者又发表了View Selection for 3D Captioning via Diffusion Ranking

而cap3d_captions.json.gz（下载地址：https://huggingface.co/datasets/tiange/Cap3D/blob/main/Objaverse_files/cap3d_captions.json.gz），根据github仓库（https://github.com/tiangeluo/DiffuRank，https://github.com/crockwell/Cap3D）、项目主页（https://cap3d-um.github.io/），和ArXiv第一版内容（https://arxiv.org/abs/2404.07984v1），这应该是第一版View Selection for 3D Captioning via Diffusion Ranking构建的数据集

Cap3D_automated_Objaverse_full.csv则是截至目前（2025.09.02），来自View Selection for 3D Captioning via Diffusion Ranking论文的最新数据集