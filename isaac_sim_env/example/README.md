example/下面的文件都是用来演示各种功能的示例脚本。

1. scene.py: 将机械臂、PartNet、Objaverse物体模型以及相机、光源、地面等环境元素加载到Isaac Sim场景中
2. robot_control.py: 演示如何使用Isaac Sim的API来控制机械臂的运动
3. task.py: 演示在Isaac Sim中实现机械臂抓取、放置方块的简单任务。为更清晰展示任务逻辑，相关工具函数被写到task_utils.py文件中
4. grasp.py: 针对任意形状物体进行抓取位姿估计。运行后会产生一个json文件，内容为采样的抓取点信息。所需的工具函数被写到grasp_init.py、grasp_utils.py文件中
5. grasp_valid.py: 验证grasp.py生成的抓取位姿的有效性。可以从json文件中复制预抓取、抓取位姿到该文件中进行验证。需要grasp_init.py中的工具函数
6. grasp_visual.py: 读取grasp.py生成的json文件，将所有抓取位姿进行可视化展示，包括物体在内。
7. articulated_valid.py: 对铰接物体进行操作。针对某铰接物体，机械臂首先移动到某link的抓取点位处，然后闭合夹爪，然后旋转关节。铰接物体中特定link的抓取点位需要事先通过grasp.py进行生成。该文件需要articulated_utils.py和grasp_init.py中的工具函数。


8. grasp_init.py: 工具函数
9. grasp_utils.py: 工具函数
10. articulated_utils.py: 工具函数
11. task_utils.py: 工具函数
12. world_transform.py: 从局部坐标系变换到世界坐标系的矩阵计算原理。