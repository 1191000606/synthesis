# # TODO：注意要还是对objaverse的urdf文件进行修改，虽然单个物体的加载没有问题，但isaacsim在对urdf文件进行读取时，采用的名称基本上都直接用的urdf中的名称，所以务必进行改名，不然会有很多麻烦
# 有哪些需要改 哪些最好改掉的？
# <robot name="cube93"> 最好改掉，至少方便看吧
# <link name="baseLink93">必须改，如果link同名会导致引用的obj文件重叠，造成物体的重复和重叠
# <visual>和<collision>最好也改掉，至少能让路径好看点，但要改这个的话，就需要把obj文件名称也改掉，注意要改两个obj。

# TODO：说到路径，现在还有一个问题，没把isaacsim中的物体路径改正确，不同物品路径有时候相交（一个物体在另一个物体下），有时候又好事，但名称后面有后缀。
# 怎么改都不行啊，完全不明白是哪冲突了，不是urdf文件名的冲突


from isaacsim import SimulationApp
kit = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

from pxr import UsdPhysics, PhysxSchema, UsdGeom, Gf, Sdf, UsdLux
import omni.kit.commands as okc
import omni.usd

# ---------- 批量导入 ----------
def load_urdf_objects(obj_ids, positions=None, scales=None, movables=None):
    stage = omni.usd.get_context().get_stage()
    n = len(obj_ids)
    # positions = positions or [(0, 0, 0)] * n
    # scales    = scales    or [(1, 1, 1)] * n
    # movables  = movables  or [True]  * n

    prim_paths = []

    # 1⃣️ 排队导入（不传 dest_path，避免之前的 Ill-formed path 问题）
    for obj_id in obj_ids:
        _, cfg = okc.execute("URDFCreateImportConfig")
        cfg.merge_fixed_joints    = False
        cfg.convex_decomp         = False
        cfg.import_inertia_tensor = True
        cfg.fix_base              = False

        urdf_path = f"/home/szwang/synthesis/data/objaverse/data/obj/{obj_id}/material.urdf"


        success, prim_path = okc.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=cfg,
            get_articulation_root=True,
        )



        print(prim_path)
        print(f"[INFO] BBBBBBBImporting {obj_id} from {urdf_path}...")
        if not success:
            print(f"[ERROR] Failed to import URDF for {obj_id}: {urdf_path}")
            prim_paths.append(None)
        else:
            prim_paths.append(prim_path)

    # 2⃣️ 统一刷新
    kit.update()

    # 3⃣️ 设置属性/变换
    for obj_id, prim_path, pos, scale, movable in zip(obj_ids, prim_paths, positions, scales, movables):
        if prim_path is None:
            continue

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARNING] prim_path invalid for {obj_id}: {prim_path}")
            continue

        # 物理开关
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(movable)
        prim.CreateAttribute("physx:rigidBody:motionEnabled", Sdf.ValueTypeNames.Bool).Set(movable)



        xform = UsdGeom.Xformable(prim)

        # 设置位置
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*pos))
                break
        else:
            xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

        # 设置缩放
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(Gf.Vec3f(*scale))
                break
        else:
            xform.AddScaleOp().Set(Gf.Vec3f(*scale))




        print(f"[INFO] Loaded {obj_id} at {pos} scale {scale} movable={movable}")

    return prim_paths

# ---------- 场景布置 ----------
stage = omni.usd.get_context().get_stage()
scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityMagnitudeAttr().Set(9.81)
PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())

okc.execute("AddGroundPlaneCommand", stage=stage, planePath="/ground",
            axis="Z", size=1500, position=Gf.Vec3f(0, 0, -0.5),
            color=Gf.Vec3f(0.5))
UsdLux.DistantLight.Define(stage, "/Sun").CreateIntensityAttr(500)

# ---------- 启动物理 ----------
omni.timeline.get_timeline_interface().play()
kit.update()

# ---------- 批量加载 ----------
load_urdf_objects(
    obj_ids   = [
        "939bce9ccaec4d5ab3404dca172d2f45",
        "3729cf312d054b9db8767c934ed13215",
        "03567b3881dc44a98ff3e6c1d449e32d",
    ],
    positions = [(5, 5, 0), (3, 2, 0), (0, 0, 0)],
    scales    = [(1, 5, 1), (2, 2, 2), (1, 1, 1)],
    movables  = [True, True, True],
)

# ---------- 主循环 ----------
while kit.is_running():
    kit.update()
