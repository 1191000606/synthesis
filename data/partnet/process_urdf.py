import os
import shutil
import xml.etree.ElementTree as ET


def fix_urdf_file(file_path):
    """
    为单个 mobility.urdf 文件创建备份，然后修复其内容。
    """
    print(f"--- 正在处理: {file_path} ---")

    # --- 步骤 1: 创建备份 ---
    backup_path = file_path + ".backup"
    try:
        # shutil.copy2 会复制文件内容和元数据
        shutil.copy2(file_path, backup_path)
        print(f"  [备份]: 成功创建备份 -> {backup_path}")
    except Exception as e:
        print(f"  [错误]: 创建备份文件失败，已跳过此文件。错误信息: {e}")
        return  # 如果备份失败，则不进行任何修改

    # --- 步骤 2: 读取并修复 URDF 文件 ---
    try:
        tree = ET.parse(file_path)
        robot_root = tree.getroot()

        # 任务 2.1: 修复命名规范 ('-' 替换为 '_')
        modified_names_count = 0
        for elem in robot_root.iter():
            if "name" in elem.attrib:
                old_name = elem.get("name")
                if "-" in old_name:
                    new_name = old_name.replace("-", "_")
                    elem.set("name", new_name)
                    modified_names_count += 1
            if "filename" in elem.attrib:
                old_filename = elem.get("filename")
                if "-" in old_filename:
                    new_filename = old_filename.replace("-", "_")
                    elem.set("filename", new_filename)
                    modified_names_count += 1
        print(f"  [命名修复]: 修正了 {modified_names_count} 个名称。")

        # 任务 2.2: 修复结构顺序 (<link> 放在 <joint> 之前)
        links = [child for child in robot_root if child.tag == "link"]
        joints = [child for child in robot_root if child.tag == "joint"]
        other_elements = [child for child in robot_root if child.tag not in ["link", "joint"]]

        if links and joints:
            # 清空并按正确顺序重组
            for child in list(robot_root):
                robot_root.remove(child)
            robot_root.extend(links)
            robot_root.extend(joints)
            robot_root.extend(other_elements)
            print("  [结构修复]: 已调整 link/joint 顺序。")
        else:
            print("  [结构修复]: 无需调整顺序。")

        # --- 步骤 3: 写回修改后的文件 ---
        # 使用 ET.indent() 美化输出格式 (需要 Python 3.9+)
        if hasattr(ET, "indent"):
            ET.indent(tree, space="    ")

        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        print(f"  [成功]: 文件已原地修改保存。")

    except Exception as e:
        print(f"  [错误]: 处理文件时发生错误: {e}")
        print(f"  [注意]: 原始文件未被修改。备份文件 '{backup_path}' 仍然可用。")


def process_partnet_dataset(base_directory):
    """
    遍历 PartNet 数据集目录，查找并处理所有 mobility.urdf 文件。
    """
    if not os.path.isdir(base_directory):
        print(f"错误：目录不存在 -> {base_directory}")
        return

    print(f"\n开始扫描目录: {base_directory}\n")
    processed_count = 0
    # 遍历 base_directory 下的所有条目，这些应该是 uid 文件夹
    for uid in os.listdir(base_directory):
        uid_path = os.path.join(base_directory, uid)

        # 确保它是一个目录
        if os.path.isdir(uid_path):
            urdf_target_file = os.path.join(uid_path, "mobility.urdf")

            # 如果 mobility.urdf 存在，则进行处理
            if os.path.isfile(urdf_target_file):
                fix_urdf_file(urdf_target_file)
                processed_count += 1

    print(f"\n扫描完成。总共处理了 {processed_count} 个 'mobility.urdf' 文件。")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 指定您的 PartNet 数据集根目录
    partnet_dataset_directory = "/home/chenyifan/Projects/synthesis/data/partnet/dataset"

    print("=" * 60)
    print("PartNet URDF 自动修复与备份脚本")
    print(f"目标目录: {partnet_dataset_directory}")
    print("=" * 60)

    process_partnet_dataset(partnet_dataset_directory)
