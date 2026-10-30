# MuJoCo 数据采集 pipeline

新增压缩格式：重开采集入口后，新的HDF5默认将双路RGB按帧编码为JPEG质量95、4:4:4（subsampling=0），保存在HDF5内，支持中断保留。回放、校验和LeRobot转换会逐帧自动解码；不需要先解压成整条大文件。旧LZF原始RGB HDF5仍兼容，已有成功回合未重写。JPEG为有损编码，“恢复”指解码为可训练RGB，不代表逐像素还原原图。状态、动作、阶段、时钟保持原数值无损保存。

已从原工程采集流程适配：`遥操作 → 双相机/实测状态/已提交动作 → HDF5 回合 → 检查与回放 → 人工标记结果 → LeRobot v3 → 双阶段边界清单`。

复用原 `data_collection/episode.py`、`isaac_adapter.py::Collector`、`stages.py` 和 `configs/stages.json`。MuJoCo 负责传感器及控制提交时机；源代码镜像保持原样。原转换脚本要求 LeRobot 0.6.2 的流式编码接口，本机已装 0.3.4 的 v3 API，因此新增兼容本机 API 的转换入口，保留原字段与验证规则。没有启动训练、上传或修改 40000 步权重。

## 1. 启动与操作

从 `D:\EAI Competition Of PPSUC` 的 PowerShell 运行：

```powershell
.\competition_local\collect_mujoco.ps1 -OperatorName teammate
```

默认输出 `competition_local/datasets/mujoco_raw/`。可用 `-Output D:\your_dataset` 改目录。无需启动 π0.5 服务；这是人工示范采集入口。默认打开 MuJoCo 和中文阶段窗口，键盘操作时让焦点处于 **MuJoCo 视口**；阶段窗口只显示状态，不发送机器人命令。`-NoPanel` 可省略中文窗口，MuJoCo HUD 仍显示录制状态。

采集窗口默认同时显示两个观测视角：左侧头部相机、右侧夹爪相机。两路均显示完整 4:3 视野，不裁边、不拉伸。点击 `Follow` 可切回第三人称，`Dual` 返回双视角；Reset 后默认回到双视角。复用原有相机，采集图像仍为各路原始 640×480 RGB。

当前采用提亮后的统一灯光配置 `mujoco_hotel/lighting_settings.json`，同时作用于预览和实际录制图像；修改后重开窗口生效。每回合文件属性保存采用的灯光设置，便于区分不同采集条件。

若 PowerShell 禁止 `.ps1`，可从任意目录直接运行项目 Python：

```powershell
& "D:\EAI Competition Of PPSUC\competition_local\.venv\Scripts\python.exe" `
  "D:\EAI Competition Of PPSUC\competition_local\mujoco_hotel\collect_mujoco.py" --operator waferen
```

| 按键或按钮 | 操作 |
|---|---|
| F1 | 新建回合，从阶段 1 开始；重复按不会覆盖当前回合 |
| F2 | 当前阶段已有至少一帧后，下一条提交动作开始使用下一阶段；到 14 停止递增 |
| F3 | 保存当前回合为 `.hdf5`，结果仍为 `unreviewed`；可提前保存，保存不代表任务成功 |
| F6 | 中断录制，保留 `.hdf5.partial`，不进入转换候选 |
| Reset 按钮 | 中断当前回合，保留 partial，复位场景；再次按 F1 才开始新回合 |
| Q / Esc / 关闭窗口 | 退出；尚未按 F3 保存的回合保留为 partial |
| 1 / 2 / 4 / 5 | 行走 / 楼梯 / 低位抓取 / 高位抓取 |
| W/S、A/D、空格 | 前后、转向、停止 |
| C / P / H | 夹爪闭合 / 张开 / 机械臂回零；机械臂控制需进入可控抓放状态 |

机械臂六关节键和 IK 键沿用旧窗口。F2 **只改人工阶段标签，不改变控制模式或机械臂动作**。14 个阶段结束后按 F3；每条独立示范前使用 Reset，避免将上回合终态当成新任务初态。

## 2. 检查与回放

```powershell
.\competition_local\mujoco_data.ps1 validate .\competition_local\datasets\mujoco_raw
.\competition_local\mujoco_data.ps1 replay .\competition_local\datasets\mujoco_raw\episode_000000.hdf5
```

回放窗口有双相机、帧滑条、时间戳和阶段英文指令，仅查看录制数据，不启动仿真。`validate -Complete` 额外检查到达阶段 14，且阶段 8 包含开始后 2 秒的旧模型兼容边界。默认检查不强迫短回合完成整场任务。

检查动作与两目标放置结果后标记：

```powershell
.\competition_local\mujoco_data.ps1 review .\competition_local\datasets\mujoco_raw\episode_000000.hdf5 -Result success -Note '两个目标均入桶，未掉落'
```

失败或不采用的回合将 `-Result` 改为 `failure` 或 `exclude`，数据仍保留。审核只修改结果属性和追加审核历史，不修改状态、动作、图像或阶段。`success` 会要求完整阶段和可用阶段边界，但**不会自动检测物体入桶**；是否成功仍由人看回放判定。

## 3. 转成 LeRobot v3

```powershell
.\competition_local\convert_mujoco.ps1 `
  -Source .\competition_local\datasets\mujoco_raw `
  -Output .\competition_local\datasets\mujoco_lerobot_v1 `
  -RepoId local/mujoco_b2w_piper
```

转换自动使用本机 `PPSUC-Ubuntu22` 中已有 `/home/eai/ppsuc/.venv/bin/python`，不要求另外启动 WSL 终端。默认仅选中 `purpose=demonstration`、人工 `result=success`、阶段完整且边界可用的回合；其他回合及原因写入 `conversion_report.json`。不上传 Hub，不训练，不混入原 Isaac 数据集。

输出目录必须不存在；失败的转换保留 `<output>.partial`，重试请换新输出名。验证成功后才发布正式目录。包含：

- `data/`：状态、动作、人工阶段和语言任务表对应的 Parquet。
- `videos/`：双路 640×480、50 fps 视频。本机 LeRobot 默认编码为 AV1（不是源脚本的 H.264）。
- `meta/`：LeRobot v3 信息、统计和回合元数据。
- `phase_boundaries.json`：复用原 OpenPI 边界构建器，首次阶段 8 的时间 + 2 秒、1e-4 秒容差。
- `conversion_report.json`：输入哈希、排除原因、逐路视频解码数、图像抽查误差、验证结果及测试数据标记。

若要显式转换尚未审核的回合，可加 `-AllowUnreviewed`；标为 failure/exclude 的回合仍不纳入。`-AllowSmoke` 仅供本管线自检，不能把其产物当成示范。训练划分应按完整回合划分，不能随机打散相邻帧当独立验证样本。

当前 40000 权重的训练配置仍指向原 Isaac 数据集。新数据的 repo_id、local_root 和 `phase_boundaries.json` 必须在后续 MuJoCo 微调配置中一致配置；还需用对应 transforms 重新计算 OpenPI 的归一化统计，尤其是六关节 delta 统计。LeRobot 的 `meta/stats.json` 不能直接替换旧权重的 OpenPI 统计。本轮交付截至采集与转换，不自动启动微调。

## 4. 数据契约

每一行记录 **执行前的状态和图像 + 本步实际提交给控制器的绝对目标**。控制器更新回零/IK/限位及 actuator ctrl 后，写入 command，再执行物理步；只有该步状态有限才提交录制帧。异常或 reset 不会把新场景状态拼到旧回合。

| 字段 | 形状 / 语义 |
|---|---|
| `state` / `command` | `[T,13]` float32：vx、wz、六关节角、归一化夹爪、四模式 one-hot |
| `images/front` / `images/wrist` | 逻辑为`[T,480,640,3]` uint8 RGB；新文件实际为`[T]`变长uint8 JPEG字节，旧文件为4维RGB；原双相机 |
| `stage_id` | `[T]` int32，从 1 开始逐级到 14 |
| `prompt` | `[14]`，原阶段英文指令，中文与版本写入元数据 |
| `timestamp` | `[T]` float64，回合内从 0 开始，每帧 0.02 秒 |
| `sim_time` | `[T]` float64，MuJoCo 绝对仿真时间；检查回合内无跳帧/复位 |
| 文件属性 | FPS、MuJoCo 版本、模型 XML 与控制器 SHA256、操作者、采集用途、审核结果、阶段版本及设置 |

vx 为根刚体质心速度在机体轴的前向分量，单位 m/s；wz 为机体轴角速度 rad/s；机械臂为绝对角 rad；夹爪按单指 0.001–0.045 m 归一化。command 存控制目标，**不存下一帧实测状态来充当动作**，也不存 16 维运动小脑内部输出。

采样按仿真时钟 50 Hz；若编码、渲染或磁盘慢，墙钟运行会变慢，不丢帧去凑实时 50 Hz。数值字段仍使用原LZF无损压缩；新图像采用独立JPEG帧，便于随机读取及partial恢复，不依赖整段视频写尾。14阶段和双阶段标签规则不变。逐帧JPEG通常不如AV1/H.264视频紧凑，但保留完整HDF5单文件及独立帧访问。

自定义脚本请使用兼容读接口（从`mujoco_hotel`目录运行）：

```python
from collection.image_storage import open_episode
with open_episode(path) as f:
    rgb = f['images/front'][0]  # 自动解码，uint8[480,640,3]
    actions = f['command'][:]   # 原数值
```

直接`h5py.File`读取新图像会得到JPEG字节，需经此接口解码。文件和相机dataset属性记录编码、质量及原始形状，格式版本为`mujoco_source_contract_v2_jpeg`。新JPEG→LeRobot视频仍有第二次图像编码，转换会保留原帧数、时钟、动作精确核验及RGB误差抽查，不能把解码成功等同于任务成功。

压缩验证：8项采集单测通过（含旧格式兼容、JPEG恢复、数值精确保留、坏帧拒绝）；118帧跨14阶段的合成测试从真实回合抽取RGB，JPEG质量95解码平均MAE约0.94/255、最大抽查MAE约1.15/255。真实WSL转换到LeRobot后双视频各118帧，数值、阶段、时间、尾padding核验通过。`datasets/mujoco_jpeg_contract*/`均为smoke_test，默认排除训练，不是成功演示。

## 5. 已完成的验证

- 6 项采集测试：执行前快照与目标复制、F2 下一动作生效、重置/待提交帧隔离、拒绝采样间隙、完整性与审核过滤、smoke 数据隔离。
- 原接口 16 项回归仍通过；修改控制循环只新增 `step_physics` 钩子，普通遥操作及 π0.5 默认物理步行为保持。
- 可见 MuJoCo 自检保存一条 **131 帧测试回合**，并保留一条 4 帧 reset 中断的 partial；中文状态窗口同时启动。阶段标签由自检脚本合成，机器人没有完成任务，`purpose=smoke_test`、`result=unreviewed`，默认训练选择器排除。
- 该测试回合已真实转换成 LeRobot v3，两路视频各解码 131 帧；Parquet 状态/动作/阶段逐值一致，首尾及边界附近 RGB 平均误差约 1.12–1.74/255；50 步动作尾部 49 步 padding 检查通过。
- 边界为阶段 8 第 14 帧起、阶段 1 第 114 帧起；现有 OpenPI `PhaseFromFrame` 实际读取后，第 113 帧的 observation phase=0、首 action phase=1，兼容检查通过。

证据目录：`competition_local/datasets/mujoco_pipeline_smoke_20261028/`、`competition_local/datasets/mujoco_pipeline_smoke_lerobot_20261028/`；日志见 `mujoco_hotel/collection_smoke_20261028.log`、`conversion_smoke_20261028.log`。这些是功能测试产物，**不是成功示范或可用成功率证据**。

复跑单元测试：

```powershell
Push-Location .\competition_local\mujoco_hotel
..\.venv\Scripts\python.exe -m collection.test_pipeline
Pop-Location
```

可见自检用 `collect_mujoco.ps1 -SmokeTest -Output <新的测试目录>`，转换时仅在测试中显式加 `-AllowSmoke -AllowUnreviewed`。

## 6. 运行边界与依赖

模式 4 的真实离地测量已修复，但低位仍约 21.5–22.1 cm，目标 23.6 cm，尚未通过 5 mm 精度验收。采集入口会如实记录这类状态，不自动把超时锁定当作成功；采示范前应先确认相应操作能稳定完成。

Windows 采集复用现有 Python 3.12/MuJoCo/Torch，额外 h5py 3.16.0 隔离在 `competition_local/collection_deps/`。WSL 转换复用现有 Python 3.11/LeRobot 0.3.4/AV/Arrow，Linux h5py 3.16.0 隔离在 `collection_deps_wsl/`，wheel 留在 `collection_wheels/`。没有升级原训练/仿真环境包。原有 HDF5 和 LeRobot 数据、源 USD、权重、阶段定义均未改写。
