from isaacsim import SimulationApp


# TODO：还没写好
# URDF import, configuration and simulation sample
kit = SimulationApp({"renderer": "RaytracedLighting", "headless": False})



import omni.kit.commands  as okc
import omni.usd
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
from isaacsim.core.prims import SingleArticulation   # 4.5 推荐接口

def load_articulated_urdf_objects(
        urdf_paths,                         # list[str]    每个 URDF 的磁盘路径
        positions=None, scales=None,        # list[tuple]  与 urdf_paths 对应
        movable_bases=None,                 # list[bool]   机器人 base 是否可动
        make_articulation=True):            # 是否返回 SingleArticulation
    """
    导入多个铰接 URDF，返回 [(prim_path, articulation|None), ...]
    仅做模型加载与基本物理属性，不创建 ground/light，也不 play Timeline。
    """
    stage     = omni.usd.get_context().get_stage()
    n         = len(urdf_paths)
    positions = positions      or [(0, 0, 0)] * n
    scales    = scales         or [(1, 1, 1)] * n
    movable_bases = movable_bases or [True]  * n

    results = []

    # ---------- 1) 依次导入 ----------
    for idx, (u, pos, scale, movable) in enumerate(
            zip(urdf_paths, positions, scales, movable_bases)):

        # 1‑a  生成不会重复的场景命名空间
        safe_name = f"art_{idx}"
        _, cfg    = okc.execute("URDFCreateImportConfig")
        cfg.merge_fixed_joints    = False
        cfg.convex_decomp         = False
        cfg.import_inertia_tensor = True
        cfg.fix_base              = False
        cfg.dest_namespace_path   = f"/{safe_name}"      # 仅影响场景 Prim 路径

        ok, prim_path = okc.execute(
            "URDFParseAndImportFile",
            urdf_path=u,
            import_config=cfg,
            get_articulation_root=True,
        )
        if not ok:
            print(f"[ERROR] URDF 导入失败: {u}")
            results.append((None, None))
            continue

        # ---------- 2) 物理 & 变换 ----------
        # prim_path 形如 /art_xxx/baseLink  → 取父级才是整个机器人根
        root_prim = stage.GetPrimAtPath(prim_path).GetParent()

        # 可动 / 不可动
        UsdPhysics.RigidBodyAPI.Apply(root_prim)\
            .CreateRigidBodyEnabledAttr(movable)
        root_prim.CreateAttribute(
            "physx:rigidBody:motionEnabled", Sdf.ValueTypeNames.Bool).Set(movable)

        xf = UsdGeom.XformCommonAPI(root_prim)
        xf.SetTranslate(Gf.Vec3d(*pos))
        xf.SetScale(   Gf.Vec3f(*scale))

        # ---------- 3) Articulation（可选） ----------
        art = None
        if make_articulation:
            art = SingleArticulation(prim_path)
            art.initialize()

        print(f"[INFO] Imported '{u}'  → {root_prim.GetPath()} "
              f"pos={pos} scale={scale} movable={movable}")

        results.append((prim_path, art))

    # 统一刷新

    kit.update()

    return results
