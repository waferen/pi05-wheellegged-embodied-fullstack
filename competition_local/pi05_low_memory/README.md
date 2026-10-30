# π0.5 12GB 显存短训练验收

状态（2026-10-25）：**已在本机RTX5070 12GB、WSL中真实完成10步双LoRA训练与本地保存。** 使用迁移的原Isaac Sim采集LeRobot数据，未恢复云端训练checkpoint。耗时116.09秒（含编译、初始化与保存），JAX峰值分配7.57GiB、内存池峰值8.01GiB，见probe_result.json。日志每步有有限loss及非零grad_norm。短探针通过不代表收敛、完整训练耗时或MuJoCo执行效果已通过。

22.5GB是OpenPI官方参考配置的需求，不是所有LoRA方案的不可突破下限。当前比赛使用JAX全量微调，不能只改batch就等同于LoRA。

## 第一阶段：先尝试 BF16 LoRA

- 微批量1，单卡，初始化使用官方pi05_base，不恢复云端训练检查点。
- PaliGemma和action expert都使用官方LoRA变体；第一轮保留官方rank，减少非必要实现改动。
- 明确冻结所有非LoRA参数，包括视觉编码器；仅调用默认get_freeze_filter不能保证视觉编码器被冻结。
- 关闭EMA，防止额外保存全模型移动平均。
- 保留源实现已有的nn.remat重计算，不把它重复开启当作新收益。
- 双相机224×224、50步动作块、14维比赛契约与归一化/阶段输入保持一致。
- 先测微批量1真实10步的前向、反向、参数更新和峰值显存；不把加载成功当作训练成功。
- 若再加梯度累积，必须在训练循环中实现并核对优化器更新频率；当前官方JAX脚本没有已确认可直接使用的累积开关，不提供虚构CLI参数。微批量1×累积32才能近似保持原有效batch32，但每次更新代价包含32次微批量前后向。

当前JAX代码的get_freeze_filter只选择llm路径的非LoRA权重，PaliGemma.img不在这个范围；因此本地专用配置将freeze_filter设为“所有非LoRA参数”。这会冻结视觉骨干和小型输入/输出投影，只训练两个语言/动作骨干的LoRA，需要独立评估策略能力。

## 如果仍超显存

依次考虑降低LoRA rank（默认2B分支16、300M分支32，rank不是现有Pi0Config的直接字段）、冻结视觉特征离线缓存、只训练动作expert，最后才考虑4-bit QLoRA与CPU卸载。缓存视觉特征会影响在线图像增强；只训expert改变可训练范围；4-bit方案需要模型实现适配和数值回归，不能直接给当前JAX脚本套用bitsandbytes开关。

截至源码核查时，OpenPI官方PyTorch路线仍注明不支持LoRA。因此采用现有JAX实现。本机PPSUC-Ubuntu22环境已通过CUDA BF16与MuJoCo测试，Linux Python位于/home/eai/ppsuc/.venv，源码独立副本位于/home/eai/ppsuc/openpi。现有Windows MuJoCo环境仍独立保留。

## 使用

`local_lora_probe.py`默认只做源码/文件预检查，不启动训练。具备独立Linux/WSL OpenPI CUDA环境后，执行 `python local_lora_probe.py --run --steps 10` 才会启动本地短探针。脚本会检查CUDA设备、使用独立输出目录，并记录设备内存统计；不连接云端。

不以缩短动作块、删掉腕部相机或更换较小模型冒充原比赛π0.5任务已适配。

来源：[OpenPI模型配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py)、[模型组件](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)、[训练循环](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train.py)、[官方PyTorch限制](https://github.com/Physical-Intelligence/openpi#pytorch-support)。
