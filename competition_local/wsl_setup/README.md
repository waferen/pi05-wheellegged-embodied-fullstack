# WSL安装与继续配置

当前状态（2026-10-25）：**环境与10步π0.5 LoRA训练均已实测通过。** 独立PPSUC-Ubuntu22，Python3.11.17、JAX0.5.3 CUDA、Torch2.7.1 CPU、MuJoCo3.17。CUDA BF16及原Piper物理检测见linux_gpu_check.json；真实模型训练与保存见../pi05_low_memory/probe_result.json，耗时116.09秒，JAX峰值分配7.57GiB。WSL自动跟进已删除，不会重复启动训练。

后台任务入口为Run-SetupJob.ps1，状态见setup_job_status.json，日志见setup_job.log。已有任务运行时不要重复启动；linux_bootstrap_status.json记录Linux安装阶段。重启已完成，不应再次执行Windows重启。

下载加速：PyPI默认使用HTTPS清华镜像，可通过PPSUC_PYPI_INDEX覆盖。若Windows已启用127.0.0.1端口代理且WSL可访问，脚本仅在本轮子进程中复用；不修改全局代理、WSL网络模式或防火墙。依赖已安装，版本见linux_requirements.lock.txt。

已完成的准备：

- 官方Ubuntu22.04.5镜像已下载，SHA-256与Microsoft WSL发行版清单一致，见ubuntu_image_verification.json。
- 发行版名PPSUC-Ubuntu22，已安装在本目录distro子文件夹，避免占用C盘主要空间。
- 用户原先没有.wslconfig。本次设置WSL上限22GB内存、12线程、8GB交换空间；交换文件位于本项目D盘。该配置对本机WSL2发行版全局生效。
- 原始迁移资产、原Windows MuJoCo环境和云端训练均不改动。Linux环境使用独立源码副本与虚拟环境。
- 独立副本保留JAX0.5.3/Flax0.10.2，并将JAX的CUDA运行库约束到12.8及以上、cuDNN9.8及以上；Torch和TorchVision配成CPU版，仅承担数据处理。该组合已通过本机GPU运算及真实短训练。
- 不在WSL里安装Linux NVIDIA显示驱动，使用现有Windows驱动595.71映射。

如后台任务已退出，需要手动续跑时，在项目目录PowerShell7运行：

```powershell
& '.\competition_local\wsl_setup\Resume-WslSetup.ps1' -RunProbe
```

脚本将依次：核对镜像→导入独立WSL2发行版→安装Ubuntu依赖→创建eai用户和独立Python3.11环境→安装OpenPI→检测CUDA/BF16与MuJoCo→显存足够时运行10步π0.5 LoRA短探针。它不会自动重启、停止其它GPU程序或连接云端。

只配环境而不启动模型探针，可省略-RunProbe。首次下载Linux依赖可能耗时较长，记录位于linux_bootstrap.log和linux_bootstrap_status.json。结果依次见linux_gpu_check.json及../pi05_low_memory/probe_result.json。

重启后检查时空闲显存约9.7GB。短探针启动前会重新检查空闲显存；不足时安全退出，等待用户已有GPU任务结束，而不会擅自关闭程序。

安装系统组件的管理员脚本和记录保留在Install-WslSystem.ps1、system_install_status.json、system_install_transcript.log。重启前授权见restart_authorization.json，重启验证见reboot_verification.json；不得据旧安装状态重复重启。

参考：[Microsoft WSL安装](https://learn.microsoft.com/en-us/windows/wsl/install)、[官方发行版清单](https://github.com/microsoft/WSL/blob/master/distributions/DistributionInfo.json)、[NVIDIA WSL GPU指南](https://docs.nvidia.com/cuda/wsl-user-guide/)。
