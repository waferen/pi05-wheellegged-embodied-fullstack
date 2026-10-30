# 基于π0.5的轮足机器人具身智能全栈开发

面向 B2W 轮足机器人与 Piper 机械臂的具身智能开发工程，包含 Isaac Sim 原工程、MuJoCo 迁移、双相机遥操作采集、HDF5 压缩、LeRobot 转换、OpenPI π0.5 微调与闭环评估。

## 资源下载

场景、机器人模型、贴图和大模型权重不进入 Git 历史。百度网盘资源包由维护者上传，**分享链接待上传完成后补充**。资源包名为“网盘上传_轮足机器人资源”。

| 文件 | 内容 |
| --- | --- |
| 01_原始场景与机器人.zip | 原始酒店场景、机器人模型、原小脑权重 |
| 02_MuJoCo场景与材质.zip | 转换后的 MuJoCo 场景、网格、贴图、USD 层和材质 |
| 03_pi05训练与推理权重.zip | run01/40000 推理检查点、最新3000更新LoRA完整检查点及统计量 |
| 04_ROS机械臂模型.zip | Piper ROS机械臂网格与模型配套文件 |
| 资源校验清单.json | 压缩包和内部文件的大小、SHA-256及目标路径 |

下载后先核对压缩包 SHA-256，再将四个 ZIP **全部解压到本仓库根目录**，与现有 `competition_local/` 合并。不要把ZIP文件名当作额外一层目录。资源包不含演示数据、虚拟环境、WSL系统盘、旧检查点或缓存。

## 目录导航

- `competition_local/cloud_snapshot/PPSUC2026/`：Isaac原工程源码、配置与采集模块。
- `competition_local/cloud_snapshot/openpi/`：OpenPI源码及本项目双阶段策略适配。
- `competition_local/mujoco_hotel/`：场景转换、控制器、双相机、π0.5推理与采集pipeline。
- `competition_local/mujoco_finetune_ep0_v2/`：最新独立单演示LoRA实验代码。
- `competition_local/pi05_low_memory/`、`wsl_setup/`：低显存配置、Linux依赖与本机安装脚本。
- 独立数据标注工具：[c4-robot-annotation](https://github.com/waferen/c4-robot-annotation)。

## Windows MuJoCo

源码快照基于 Windows Python3.12，实际安装版本见 `competition_local/requirements.lock.txt`。先安装相应Python并恢复资源：

```powershell
cd competition_local
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\.venv\Scripts\python.exe -m pip install h5py msgpack
.\.venv\Scripts\python.exe -m pip install -e .\cloud_snapshot\openpi\packages\openpi-client
powershell -ExecutionPolicy Bypass -File .\view_hotel.ps1
```

采集双相机演示（头部、夹爪）：

```powershell
powershell -ExecutionPolicy Bypass -File .\collect_mujoco.ps1 -OperatorName teammate
```

F1开始、F2下一阶段、F3保存，F6中断保留partial。新HDF5图像为JPEG95，状态/动作保持无损，回放与转换自动解码。完整说明见 [采集文档](competition_local/mujoco_hotel/collection/README.md)。

## π0.5 训练与推理

π0.5使用WSL/Linux环境，JAX CUDA与Windows仿真分离。环境及依赖参考 [WSL说明](competition_local/wsl_setup/README.md) 与 [低显存说明](competition_local/pi05_low_memory/README.md)。源码中 `/home/eai/ppsuc`、发行版名称和本机路径需要按部署环境调整；这些脚本是本机实验快照，不是零配置安装器。`wsl_setup`中的系统安装/重启脚本不应在已有环境直接执行。

先审核成功演示，再转换成LeRobot。最新实验的 `experiment.py`、`prepare.py`、`train.py`、`serve.py`分别负责配置、归一化和契约检查、训练、服务。`run.py`会实际启动训练；本发布包不包含演示数据，需自行采集并修改实验的数据路径、帧数与阶段边界。默认不能直接复用单回合的17674帧断言。

原40000推理服务入口为 `competition_local/serve_pi05.ps1`，MuJoCo入口为 `view_pi05.ps1`。最新LoRA服务使用 `mujoco_finetune_ep0_v2/serve.py`，默认端口8767。启动前核对检查点、统计量和端口，避免把原权重与LoRA统计量混用。

## 已知结果与限制

- 原run01/40000来自Isaac遥操作数据训练，已迁移并实际读取验证。
- 单条成功MuJoCo演示完成3000次双LoRA更新，batch1、BF16冻结主权重、关闭EMA；是LoRA，不是4bit QLoRA。
- 420秒自主MuJoCo评估未完成抓取入仓；机器人未可靠退出楼梯模式。
- 离线检查显示维持已有模式的准确率高，但模式切换和双阶段推进仍有缺陷。训练损失和普通帧准确率不能代表任务成功率。
- 当前只有单演示拟合与有限闭环测试，尚未验证泛化或完整任务成功。

## 来源与许可

保留PPSUC2026、OpenPI及ROS依赖各自LICENSE/NOTICE；这些组件遵循各自许可。场景、贴图和第三方模型的使用权以其原始来源条款为准。仓库未对第三方资源另授新许可证。
