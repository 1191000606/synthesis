import os
import xml.etree.ElementTree as ET


def process_urdf_files(root_dir):
    """
    遍历指定目录下的所有子文件夹，找到 mobility.urdf 文件，并修改其中的 link 和 joint 名称。

    参数:
    root_dir (str): 数据集的根目录路径。
    """
    if not os.path.exists(root_dir):
        print(f"错误: 指定的目录 '{root_dir}' 不存在。")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前目录是否是 UID 子文件夹
        if os.path.basename(dirpath).isdigit() and "mobility.urdf" in filenames:
            urdf_path = os.path.join(dirpath, "mobility.urdf")

            print(f"正在处理文件: {urdf_path}")

            try:
                tree = ET.parse(urdf_path)
                root = tree.getroot()

                # 1. 从 <robot> 标签中提取 UID
                robot_name = root.get("name")
                if not robot_name or "_" not in robot_name:
                    print(f"警告: 文件 {urdf_path} 中的 <robot> name 格式不正确，跳过。")
                    continue

                uid = robot_name.split("_")[-1]
                print(f"找到 UID: {uid}")

                # 2. 修改所有的 <link> 标签
                for link in root.findall("link"):
                    original_name = link.get("name")
                    if original_name:
                        new_name = f"{original_name}_{uid}"
                        link.set("name", new_name)
                        print(f"  - 修改 <link> 名称: {original_name} -> {new_name}")

                # 3. 修改所有的 <joint> 标签及其子标签
                for joint in root.findall("joint"):
                    # 修改 <joint> 的 name
                    joint_original_name = joint.get("name")
                    if joint_original_name:
                        new_joint_name = f"{joint_original_name}_{uid}"
                        joint.set("name", new_joint_name)
                        print(f"  - 修改 <joint> 名称: {joint_original_name} -> {new_joint_name}")

                    # 修改 <child> 标签的 link
                    child_link = joint.find("child")
                    if child_link is not None:
                        child_original_link = child_link.get("link")
                        if child_original_link:
                            new_child_link = f"{child_original_link}_{uid}"
                            child_link.set("link", new_child_link)
                            print(f"  - 修改 <child> 链接: {child_original_link} -> {new_child_link}")

                    # 修改 <parent> 标签的 link
                    parent_link = joint.find("parent")
                    if parent_link is not None:
                        parent_original_link = parent_link.get("link")
                        if parent_original_link:
                            new_parent_link = f"{parent_original_link}_{uid}"
                            parent_link.set("link", new_parent_link)
                            print(f"  - 修改 <parent> 链接: {parent_original_link} -> {new_parent_link}")

                # 4. 将修改后的内容写回文件
                # 注意：ElementTree 的 write 方法会添加 XML 声明，为了保持格式一致，我们手动添加
                ET.indent(tree, " " * 4)  # 使输出格式化，更易读
                tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

                print(f"文件 {urdf_path} 处理完成并已保存。\n")

            except ET.ParseError as e:
                print(f"错误: 解析文件 {urdf_path} 失败 - {e}")
            except Exception as e:
                print(f"处理文件 {urdf_path} 时发生未知错误: {e}")


if __name__ == "__main__":
    # 请将这里的路径替换为你的数据集根目录
    dataset_path = "/home/chenyifan/Projects/synthesis/data/partnet/dataset"

    print(f"开始处理数据集目录: {dataset_path}\n")
    process_urdf_files(dataset_path)
    print("\n所有文件处理完成。")
