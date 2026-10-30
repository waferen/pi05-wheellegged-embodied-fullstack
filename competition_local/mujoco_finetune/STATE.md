# 单演示 MuJoCo π0.5 微调

**本轮目标已完成，停止重复训练/转换/评估。结果：500更新完成，420秒自主MuJoCo试验未完成抓取入仓。详见 `结果说明.md`。**

最终有效评估目录 `evaluation_pilot500_cached/`；此前 `evaluation_pilot500/`、`evaluation_pilot500_retry/` 是内存错误失败，`evaluation_pilot500_final/` 为服务等待后中断，不代表最终结果。500步完整检查点已内容校验导出到 `portable_checkpoint499/`，原40000权重未改。下面“运行中”段落为历史过程记录，当前事实以本段、结果说明和status.json完成字段为准。自动跟进mujoco-0-5应删除，goal应标complete。

用户本轮明确授权：采用最新成功示范试微调后在 MuJoCo 推理；注意 Codex 额度，用 goal 和恢复后续接。覆盖此前“训练仅用户手动启动”的约束，仅针对本实验。

## 输入与限制

- 唯一输入 `D:/EAI Competition Of PPSUC/competition_local/datasets/mujoco_raw/episode_000003.hdf5`，用户确认成功；17601 帧，352.02 秒，50Hz，双路640×480，14阶段；阶段8+2秒边界9409。
- 000000/000001/000002 都是 partial，不得使用；不混入其它示范、smoke 或原Isaac数据。
- 从已迁移主模型 `cloud_snapshot/openpi/checkpoints/pi05_b2w_piper/run01/40000/params` 初始化，非官方base，也非历史LoRA探针。
- 保留40000原权重，另存微调LoRA和优化器检查点；不开云端付费资源，不上传数据。
- 单回合只能报告同场景试验，不建立独立泛化成功率。

## 运行与后续

- WSL `PPSUC-Ubuntu22` 用户 `eai`，Python `/home/eai/ppsuc/.venv/bin/python`；源训练库 `/home/eai/ppsuc/openpi`。通过WSL命令需沙盒提升，已成功；普通沙盒会E_ACCESSDENIED。
- 本机RTX5070 12GB，初查Windows/桌面约占3GB，无现有训练/策略服务。选择双LoRA、microbatch1、冻结基础权重、关闭EMA。
- 按原输入变换计算专用归一化，固定14维动作和双阶段协议。训练、推理必须使用同一组新统计量。
- 先跑短期微调并保存，再做可见MuJoCo闭环评估；不得将动作回放冒充策略闭环成功。
- goal 已建立，完成前保持active。账户初查5小时已用69%，周已用14%。任一窗口达到95%后保存必要进度，不再开始新步骤，额度恢复后接续。不自动购买额度或用重置次数。
- 本聊天自动跟进ID `mujoco-0-5`，每小时续接；先读本文件和status.json、核查进程/输出避免重复，完成后停用。

实时进展见 status.json 与各阶段日志；任何错误保留并修复，不以目录存在当作训练完成。

## 已完成与运行中

- 已按用户本聊天的成功确认给唯一完整回合追加 success 审核记录；原state/command/图像未改动。
- 转换完成：`../datasets/mujoco_success_ep3_v1`，repo_id `local/mujoco_success_ep3`；双AV1视频各17601帧、50Hz，数值/文本/尾部padding/抽查RGB通过，未包含其它回合。原文件审核后SHA256 `7f4a247ed79fc84ef2dae92d4b8c60b2229cf2042c72799f94dc274660817ad5`。
- `prepare_result.json` passed=true：使用实际DeltaActions处理全部17601×50动作，独立统计13维物理量；阶段在归一化后追加。真实OpenPI数据变换抽查0/4000/9408/9409/17600帧均通过。
- 训练进程见status.json的train.pid，目标500更新、batch1、双LoRA、warmup25、peak_lr5e-5、每50步保存、最多保留1个恢复检查点；配置 `experiment.py`，实际记录 `train_config.json`；日志 `train.log`。`train_result.json` passed=true 才算完成，不能只看目录。当前已有GPU可用、数据加载器通过，开始加载40000权重。
- 训练完成后通过WSL执行 `job.py serve`（8766，避免覆盖历史8765），检查serve.log确实就绪，再用Windows虚拟环境运行 `evaluate.py`。它会在原可见双相机GUI中自主推理，默认420仿真秒；严重跌倒或60秒底座/机械臂均无进展时停止，保留 `evaluation_pilot500/closed_loop.mp4`、result.json和轨迹。不手动代操作；若用户接管则报告中断，不判成功。
- 评估成功辅助条件仅为两个easy目标释放后质心在原仓体区域持续4秒；应结合视频判定，不能当作多回合成功率。没有成功也应如实交付本次微调与推理结果，不无界重训。
- 后台启动器：`wsl -d PPSUC-Ubuntu22 -u eai --exec /home/eai/ppsuc/.venv/bin/python '/mnt/d/EAI Competition Of PPSUC/competition_local/mujoco_finetune/job.py' <prepare|train|serve>`。它保存PID并避免同job重复启动。首次尝试bash nohup没有启动任何任务；实际使用Python启动器，旧convert.sh未使用。
- 当前5小时额度已用约80%，周15%；后续先复核账户额度，达到95%只写必要进度后停止新工作。已提交的500步本地训练有界运行并保存，不调用OpenAI API。


## 最终收尾状态

Goal已标complete；本任务8766策略服务已停止，释放GPU。累计目标统计163600 tokens、约65分钟（不是账户额度百分比）。账户额度查询后续超时；未购买或使用重置。已调用automation_update删除mujoco-0-5，但工具未响应，未确认删除成功；未来心跳只需重试删除自动跟进并保持安静，不得重启训练或推理。


自动跟进清理已完成：automation_update明确确认mujoco-0-5已删除，不再每小时唤醒；无需用户手动处理。此前删除超时记录仅为历史。
