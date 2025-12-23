import os
import shutil


def process_partnet_dataset(base_dir):
    """
    处理 PartNet 数据集中的 .obj 和 .mtl 文件。

    Args:
        base_dir (str): 数据集的根目录，例如 '/home/chenyifan/Projects/synthesis/data/partnet/dataset'。
    """
    print(f"正在处理数据集目录: {base_dir}")

    # 遍历所有以数字命名的子目录
    for uid in os.listdir(base_dir):
        uid_dir = os.path.join(base_dir, uid)

        # 检查是否是有效的uid目录（只包含数字且为目录）
        if not os.path.isdir(uid_dir) or not uid.isdigit():
            continue

        print(f"\n--- 正在处理 UID: {uid} ---")
        textured_objs_dir = os.path.join(uid_dir, "textured_objs")

        # 检查 textured_objs 目录是否存在
        if not os.path.exists(textured_objs_dir):
            print(f"警告: 目录 {textured_objs_dir} 不存在，跳过。")
            continue

        # 1. 备份 textured_objs 目录
        textured_objs_backup_dir = os.path.join(uid_dir, "textured_objs_backup")
        if os.path.exists(textured_objs_backup_dir):
            print(f"备份目录 {textured_objs_backup_dir} 已存在，跳过备份。")
        else:
            try:
                shutil.copytree(textured_objs_dir, textured_objs_backup_dir)
                print(f"成功创建备份目录: {textured_objs_backup_dir}")
            except Exception as e:
                print(f"备份失败: {e}")
                continue

        # 2. 遍历并修改文件
        for filename in os.listdir(textured_objs_dir):
            file_path = os.path.join(textured_objs_dir, filename)

            # 检查文件是否为 .obj 或 .mtl
            if os.path.isfile(file_path) and (filename.endswith(".obj") or filename.endswith(".mtl")):

                # 3. 重命名文件 (如果有需要)
                new_filename = filename.replace("-", "_")
                if new_filename != filename:
                    new_file_path = os.path.join(textured_objs_dir, new_filename)
                    os.rename(file_path, new_file_path)
                    print(f"已重命名文件: {filename} -> {new_filename}")
                    file_path = new_file_path  # 更新路径以便后续处理

                # 4. 修改文件内容
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 替换内容中的文件名
                    new_content = content.replace("original-", "original_").replace("new-", "new_")

                    if new_content != content:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"已更新文件内容: {new_filename}")

                except UnicodeDecodeError:
                    print(f"警告: 无法以 UTF-8 编码读取文件 {file_path}，跳过内容修改。")
                except Exception as e:
                    print(f"处理文件 {file_path} 时发生错误: {e}")


if __name__ == "__main__":
    # 定义你的数据集根目录
    dataset_root = "/home/chenyifan/Projects/synthesis/data/partnet/dataset"

    # 确保根目录存在
    if not os.path.isdir(dataset_root):
        print(f"错误: 找不到数据集根目录 {dataset_root}")
    else:
        process_partnet_dataset(dataset_root)
