# WSL本地训练环境配置

用户已授权配置WSL并尝试π0.5低显存训练。当前迁移任务已经完成，云端仍训练；本阶段不连接或操作云端。

## 当前状态（2026-10-25约13:05）

### 重启后续跑（2026-10-25约13:20）

#### 最终实测（2026-10-25 13:42）

- WSL环境配置完成，linux_gpu_check.json证明JAX0.5.3 CUDA BF16矩阵乘与MuJoCo3.17原Piper模型物理推进均通过；Torch2.7.1为CPU版。
- π0.5双LoRA/batch1/BF16/冻结全部非LoRA参数/无EMA，真实训练10步完成并保存本地检查点；probe_result.json runtime_verified=true，elapsed_seconds=116.093373531，JAX peak_bytes_in_use=8127609088（7.57GiB），peak_pool_bytes=8596226048（8.01GiB）。这仅证明10步可运行，不代表收敛或完整训练性能。
- 初次短探针发现PhaseFromFrame按新实验assets目录读取phase_boundaries.json；已复制原阶段边界与归一化JSON到独立探针assets目录，内容不改。DrvFS跨系统复制不保留POSIX时间元数据，使用copyfile避免copy2的utime权限错误。未修改原始数字资产。
- 后台wrapper PID11780已正常退出，setup_job_status.json phase=complete。WSL自动跟进wsl已删除，避免重复训练。后续当前用户请求是酒店与机器狗MuJoCo适配，代码另存mujoco_hotel。

- 已现场确认Windows启动时间为2026-10-25 13:11:11，晚于授权文件中的旧启动时间；不需要再次重启。
- WSL服务可以访问，当前没有既有发行版。已启动独立后台安装任务Run-SetupJob.ps1，状态见setup_job_status.json，日志见setup_job.log和linux_bootstrap.log。
- 首次后台启动使用Windows PowerShell5时无法加载Get-FileHash；该进程已退出。已改用Codex自带PowerShell7重新启动，未重复创建存活任务。
- 后续继续以实际linux_gpu_check.json和pi05_low_memory/probe_result.json为验收，尚不能宣称GPU或模型训练通过。
- Ubuntu已导入成功并运行于WSL2，Linux中nvidia-smi识别RTX5070/驱动595.71/总显存12227MiB/空闲9898MiB。系统包安装完成。
- files.pythonhosted.org下载uv实测几KiB/s，已只停止匹配的本地pip安装进程后切换HTTPS清华PyPI镜像；同一17.2MB轮子下载达到8.7MB/s。后续脚本默认使用该镜像（PPSUC_PYPI_INDEX可覆盖），CPU Torch专用源保持独立。旧后台任务已退出后才启动新任务，最新PID以setup_job_status.json为准。
- Python官方运行时直连同样慢。确认用户已启用的Windows代理127.0.0.1:7890原本监听通配地址，从WSL通过默认网关访问成功（1MiB测速约1.6MB/s）；network_env.sh仅为当前子进程复用该代理，未修改代理、网络模式或防火墙。清华镜像直连。Python3.11.17已安装，uv已解析254包并开始下载CUDA/JAX/CPU Torch依赖。
- 用户随后要求打开MuJoCo查看迁移效果：已用原Windows venv的pythonw启动replay.py --viewer --episodes 10000，真实图形子进程26584，窗口标题MuJoCo : PPSUC Piper local joint reaching，Responding=true，无stderr。不要擅自关闭此用户查看窗口。它展示已训练的Piper两关节到点任务，完整酒店尚未适配。
- WSL后台任务继续运行（本轮最后核实wrapper PID27488，Linux uv sync PID1313）；剩余大型CUDA包仍下载中。自动跟进应复用该任务，不能重复启动，也不需要重启。

- 用户已明确回复“现在可以重启，我已保存工作”。授权与Windows重启前启动时间保存在restart_authorization.json。本轮将只发出一次Windows重启；后续不得因看到旧状态而自行重复重启。
- 已设置本聊天15分钟自动跟进。Codex恢复运行后，比较Windows启动时间；确实重启且额度允许时运行Resume-WslSetup.ps1 -RunProbe。若已有WSL/初始化任务存活，复用该任务，不重复启动。完成或需要用户操作时才通知，最终删除此WSL跟进。

- 管理员脚本已真实执行成功：VirtualMachinePlatform由Disabled变为Enabled，RestartNeeded=true。WSL3.0.3、内核6.18.54.1-1、WSLg1.0.79已安装。
- Ubuntu22.04.5官方镜像343.9MiB已下载并校验，SHA256=4499c4fe257f2fc83145b429ce211a0a43fd590e70d6261ede616210947d9f8f，与Microsoft发行版清单一致。
- WSL发行版导入已尝试，因尚未重启返回HCS_E_SERVICE_NOT_AVAILABLE。没有擅自重启、注销或卸载发行版。
- 当前硬门槛：用户保存其它应用工作后确认何时重启Windows。尚未注册成功的发行版不能当成已运行环境。
- 重启后入口：wsl_setup/Resume-WslSetup.ps1 -RunProbe；脚本与README均已准备，PowerShell语法、Python脚本编译及独立TOML生成已验证；Linux端执行尚未验证。
- .wslconfig新建于C:\Users\28358，限22GB内存、12线程、8GB D盘交换；此前没有该文件。没有改动旧配置。
- 当前迁移原项目CUDA依赖锁到12.6/cuDNN9.5，对Blackwell需要独立环境适配。configure_linux_project.py只修改Linux源码副本，JAX/Flax不变，CPU Torch2.7.1与CPU TorchVision0.22.1配套，并约束JAX CUDA>=12.8/cuDNN>=9.8。依赖解析与实际GPU必须现场验证，不能预称兼容。
- 本次修改local_lora_probe.py，支持PPSUC_OPENPI_ROOT与PPSUC_RUN_ROOT，原始数据仍从D盘读取，Linux环境与训练输出可放WSL ext4。
- 若重启后显存仍只有约4.8GB空闲，run_lora_probe.sh会安全退出并记录提示，不得擅自杀用户GPU任务。

- 系统：Windows11家庭版26200，Intel i7-14700HX，约32GB RAM；HypervisorPresent=true。
- GPU：RTX5070 12GB，Windows驱动595.71。开始检查时显存空闲约4.8GB；不得擅自关闭用户程序来释放显存。
- 当前普通终端不是管理员，WSL尚未安装。
- `Install-WslSystem.ps1`已准备：只启用VirtualMachinePlatform并通过Microsoft官方wsl命令安装WSL包，不自动重启。管理员授权由Windows UAC交给用户。
- 安装日志：system_install_transcript.log；状态：system_install_status.json。
- 若系统要求重启，先完成发行版/依赖脚本准备并保存状态，再请用户确认重启时间；不要自行重启Windows。
- WSL中使用Windows现有NVIDIA驱动映射，不安装Linux显示驱动。
- 计划：Ubuntu22.04独立环境、Python3.11、项目固定依赖、GPU/JAX检测，再运行pi05_low_memory/local_lora_probe.py真实短探针。禁止把配置通过或GPU可见当作π0.5训练已通过。
