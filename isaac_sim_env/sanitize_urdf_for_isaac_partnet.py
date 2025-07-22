#!/usr/bin/env python3
# usage: python sanitize_urdf_for_isaac_partnet.py /home/szwang/synthesis/data/dataset/168/mobility.urdf
# 此脚本目前已经实现的功能
# 修改urdf中的各类名称，将-换成_
# 将obj文件中引用的mtl名称-也换成_
# mtl文件中也会修改newmtl和map_Kd行中的名称
# 修改obj和mtl文件名中的-为_

# 后面可以再加上质量信息

import argparse, os, shutil, pathlib, re, xml.etree.ElementTree as ET

INVALID_CH = "-"
REPL_CH    = "_"
OBJ_DIR    = "textured_objs"       # 放置 mesh 的文件夹名

# ---------- 基础工具 ----------
def sanitize_token(s: str) -> str:
    return s.replace(INVALID_CH, REPL_CH)

def rename_file(old: pathlib.Path, dry=False) -> pathlib.Path:
    """把文件名中的 '-' 改 '_'，返回新路径（若没变则返回原 Path）"""
    new_name = sanitize_token(old.name)
    if new_name == old.name:
        return old
    new = old.with_name(new_name)
    if dry:
        print(f"[DRY] rename {old.name} -> {new_name}")
    else:
        print(f"[RENAME] {old.name} -> {new_name}")
        old.rename(new)
    return new

# ---------- ① 批量改名 + 修 obj & mtl 内容 ----------
def process_obj_and_mtl(root_dir: pathlib.Path, dry=False):
    obj_root = root_dir / OBJ_DIR
    if not obj_root.is_dir():
        print(f"[WARN] mesh dir not found: {obj_root}")
        return

    # 1) 先把 .mtl 文件本身改名（material-0.mtl -> material_0.mtl）
    mtl_map = {}
    for mtl in obj_root.glob("**/*.mtl"):
        new_path = rename_file(mtl, dry)
        mtl_map[mtl.name] = new_path.name   # 旧名 → 新名

        # --- 新增：改 mtl 内部 newmtl / map_Kd 行 ---
        if dry:
            continue

        lines = new_path.read_text().splitlines()
        changed = False
        for i, line in enumerate(lines):
            low = line.lower()
            if low.startswith("newmtl"):
                head, old_name = line.split(maxsplit=1)
                new_name = sanitize_token(old_name)
                if new_name != old_name:
                    lines[i] = f"{head} {new_name}"
                    changed = True
            elif low.startswith("map_kd"):
                head, old_tex = line.split(maxsplit=1)
                new_tex = sanitize_token(old_tex)
                if new_tex != old_tex:
                    lines[i] = f"{head} {new_tex}"
                    changed = True
        if changed:
            new_path.write_text("\n".join(lines))
            print(f"[FIX]   newmtl/map_Kd refs in {new_path.name}")

    # 2) 处理 .obj：改名 + 修 mtllib/usemtl 引用
    mtllib_pat = re.compile(r"^(mtllib\s+)(.+)$", flags=re.IGNORECASE)
    usemtl_pat = re.compile(r"^(usemtl\s+)(.+)$", flags=re.IGNORECASE)

    for obj in obj_root.glob("**/*.obj"):
        new_obj_path = rename_file(obj, dry)
        if dry:
            continue

        lines = new_obj_path.read_text().splitlines()
        changed = False
        for i, line in enumerate(lines):
            for pat in (mtllib_pat, usemtl_pat):
                m = pat.match(line)
                if m:
                    prefix, name = m.groups()
                    fixed_name = sanitize_token(name)          # material-7 → material_7
                    fixed_name = mtl_map.get(name, fixed_name) # 若 mtl 文件名也改了
                    if fixed_name != name:
                        lines[i] = f"{prefix}{fixed_name}"
                        changed = True
        if changed:
            new_obj_path.write_text("\n".join(lines))
            print(f"[FIX]   mtllib/usemtl refs in {new_obj_path.name}")


# ---------- ② 清洗 URDF ----------
def sanitize_mesh_path(mesh_path: str, urdf_dir: pathlib.Path, dry=False):
    rel = pathlib.Path(mesh_path)
    full = rel if rel.is_absolute() else urdf_dir / rel
    if not full.exists():
        print(f"[WARN] mesh not found: {full}")
        return str(rel).replace(INVALID_CH, REPL_CH)

    new_full = rename_file(full, dry=dry)
    return str(rel.with_name(new_full.name))

def sanitize_urdf(urdf_path: pathlib.Path, dry=False):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent

    for elem in root.iter():
        if "name" in elem.attrib:
            old, new = elem.attrib["name"], sanitize_token(elem.attrib["name"])
            if new != old:
                print(f"[NAME] {old} -> {new}")
                elem.set("name", new)

    for mesh in root.iter("mesh"):
        old = mesh.get("filename", "")
        if not old:
            continue
        new = sanitize_mesh_path(old, urdf_dir, dry=dry)
        if new != old:
            print(f"[MESH] {old} -> {new}")
            mesh.set("filename", new)

    if dry:
        out = urdf_path.with_suffix(".sanitized.xml")
        tree.write(out, encoding="utf-8", xml_declaration=True)
        print(f"[DRY] wrote preview: {out}")
    else:
        bak = urdf_path.with_suffix(urdf_path.suffix + ".bak")
        shutil.copy2(urdf_path, bak)
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
        print(f"[DONE] sanitized; backup → {bak}")

# ---------- ③ 主入口 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true", help="preview only")
    args = ap.parse_args()

    root_dir = args.urdf.parent
    process_obj_and_mtl(root_dir, dry=args.dry_run)
    sanitize_urdf(args.urdf, dry=args.dry_run)

if __name__ == "__main__":
    main()
