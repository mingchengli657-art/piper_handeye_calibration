# Piper Hand–Eye Calibration

面向 **Piper X + Intel RealSense D405** 的眼在手上标定工具。相机刚性固定在机械臂末端，ChArUco 板固定在工作空间，通过多组机械臂姿态求出相机到 TCP/法兰的外参。

从机器人比赛项目中提取，包含 **标定板检测 → 时间戳配对采集 → 样本筛选 → 五种算法求解 → 独立样本验证 → ROS 2 静态 TF 发布**。采集和发布脚本不发送机械臂运动指令；机械臂驱动和相机驱动由使用者单独启动。

## 环境

已在本次整理环境中验证离线求解：Python 3.10、OpenCV 4.5.4、NumPy、SciPy、PyYAML；ROS 2 Humble 模块导入和命令行检查通过。尚未在本次整理中连接硬件重新采集。原始工程使用 ROS 2 Humble。

### ROS 2 采集环境（Ubuntu 22.04 / Humble）

先安装 ROS 2 Humble，再安装本项目需要的系统包：

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy python3-scipy python3-yaml \
  ros-humble-cv-bridge ros-humble-tf2-ros ros-humble-geometry-msgs ros-humble-sensor-msgs
source /opt/ros/humble/setup.bash
```

使用系统 `python3`，OpenCV 必须带 `cv2.aruco` 和 GUI 支持。ROS 采集环境优先使用系统包，避免混用 pip OpenCV 与系统 `cv_bridge`。D405 和 Piper 驱动不包含在本仓库；只要能发布下面的标准消息接口即可接入。

### 仅离线复现（不需要 ROS 或硬件）

解压或克隆本仓库，进入仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python scripts/solve_handeye.py --metadata-dir examples/metadata/train
python scripts/validate_result.py --metadata-dir examples/metadata/validation
python -m unittest discover -s tests -v
```

生成 `results/handeye_result.yaml`、`results/validation_report.yaml` 和逐样本 CSV。依赖文件针对 Python 3.10 和原始 OpenCV 4.x API；其他版本需要自行验证兼容性。

## 真实样本复现

仓库附带原始采集中的 93 份 JSON 元数据，不包含约 219 MB 的完整图像数据：

| 数据集 | 原始样本编号 | 默认筛选后 |
|---|---|---|
| 求解集 | 1–78 | 43 组 |
| 独立验证集 | 79–93 | 10 组 |

默认要求至少 88 个 ChArUco 角点、图像与机器人位姿时间差不超过 0.05 秒。原始最终结果单独保存在 `examples/reference/`，不会作为新设备的默认标定结果。

复现得到的相机到 TCP 平移约为 `[-12.303, -73.976, 46.944] mm`，独立验证平移 RMS **1.861 mm**、最大 **3.334 mm**；旋转 RMS **0.185°**、最大 **0.305°**。

这些是固定板在基座坐标系中的**一致性指标**，不是绝对定位精度或抓取精度，也不保证在其他设备上达到相同性能。没有随包提供原始图像，因此示例支持从已估计位姿开始复现，不能重新检测历史图像角点。

## 坐标约定

`A_T_B` 表示将 B 坐标系中的点变换到 A 坐标系；长度单位为米，旋转向量为弧度，四元数顺序为 `[x, y, z, w]`。

```text
base_T_tcp × tcp_T_camera × camera_T_target = base_T_target
```

- 输入 `base_T_tcp`：机械臂反馈的 TCP 在基座坐标系中的位姿。
- 输入 `camera_T_target`：ChArUco 板在彩色相机光学坐标系中的位姿。
- 输出 `tcp_T_camera`：相机光学坐标系到 TCP 的固定变换。

原始采集时 TCP 偏移为零，因此 TCP 与法兰等价；原示例 TF 父坐标系使用 `flange_link`。如果你的驱动设置了工具偏移，必须使用与采集反馈一致的 TCP 定义。相机光学坐标系不能直接替换为相机机身坐标系。

OpenCV 的输入方向对应 `gripper2base`、`target2cam`，返回 `cam2gripper`，参见 [OpenCV 手眼标定接口](https://docs.opencv.org/3.4.15/d9/d0c/group__calib3d.html)。

## 从头标定

以下命令均在仓库根目录运行。先启动设备驱动，并确认接口：

| 数据 | 默认话题 | 消息类型 |
|---|---|---|
| 彩色图像 | `/camera/d405/color/image_raw` | `sensor_msgs/msg/Image` |
| 相机内参 | `/camera/d405/color/camera_info` | `sensor_msgs/msg/CameraInfo` |
| TCP 位姿 | `/feedback/tcp_pose` | `geometry_msgs/msg/PoseStamped` |

图像和位姿必须使用同一时间基准、非零时间戳。反馈必须是米制的 `base_T_tcp`，不是关节角或毫米制位姿。程序按时间戳匹配，不执行 TF 查询来纠正错误坐标系。默认订阅使用 reliable QoS；若驱动只提供 best-effort，需调整订阅或驱动 QoS 后再采集。

### 1. 检查标定板

默认 `config/board.yaml` 对应原始实物板：12 × 9 个方格、15 mm 方格边长、11.25 mm ArUco 边长、`DICT_5X5_100`，共 88 个内角点。必须测量实际打印尺寸；随意缩放打印会改变平移尺度。本包没有原始打印板文件。

```bash
python3 scripts/charuco_detector.py --board config/board.yaml
```

图像窗口应显示角点和坐标轴。`s` 保存检测图，`q` 退出。换板时修改配置，并同步修改后续 `--min-corners`；采集器至少要求 20 个角点，因此小于 20 个内角点的板不在当前支持范围内。新 OpenCV 的旧板模式见 [ChArUco 板兼容说明](https://docs.opencv.org/4.13.0/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html)。

### 2. 采集求解样本

固定标定板和相机支架。用独立的机械臂控制方式改变姿态，每次停稳后按 `c` 保存，`q` 退出。采集覆盖不同位置、倾角和至少两个非平行旋转轴的姿态；不要只做平移或在同一轴上旋转。

```bash
python3 scripts/collect_samples.py \
  --board config/board.yaml \
  --output-dir data/train \
  --sync-limit 0.05
```

可通过 `--image-topic`、`--camera-info-topic`、`--pose-topic` 替换默认话题。`--count 30` 表示本次保存 30 组后退出，默认不限数量。再次运行会接续已有编号。采集器允许保存至少 20 角点的观测，求解默认只采用完整 88 角点观测，因此采集数不等于有效求解数。

输出 `images/`、`overlays/`、`metadata/`。采集器拒绝时间差超限、零图像时间戳、超过一秒未更新的图像及重复图像；它不自动判断机械臂是否停稳。

### 3. 求解外参

```bash
python3 scripts/solve_handeye.py \
  --metadata-dir data/train/metadata \
  --min-corners 88 --max-dt 0.05 \
  --output results/handeye_result.yaml
```

比较 Tsai、Park、Horaud、Andreff、Daniilidis，默认采用 Park；Park 失败时使用第一个有效方法，并在报告中记录。默认方法不是按最小误差自动选出的。报告保存采用/排除样本、排除原因、变换矩阵、平移、四元数及各方法一致性指标。

至少需要三组有效样本，但三组只是数学下限。程序会拒绝明显退化的旋转和无效矩阵，不能替代姿态覆盖质量检查。可用 `--exclude-sample-ids 10 11 12` 排除保留验证的样本。

### 4. 独立验证

保持板的位置不动，采集一批未用于求解的新姿态：

```bash
python3 scripts/collect_samples.py --output-dir data/validation --sync-limit 0.05
python3 scripts/validate_result.py \
  --metadata-dir data/validation/metadata \
  --result results/handeye_result.yaml \
  --min-corners 88 --max-dt 0.05 \
  --output results/validation_report.yaml
```

验证生成 YAML 汇总和 CSV 逐样本误差。独立目录的编号从 1 重新开始，编号重叠提示只表示需要检查数据来源，不代表样本必然重复。使用同一目录时可用 `--min-sample-id` / `--max-sample-id` 选择验证区间，并在求解时显式排除这些编号。

求解报告的旋转误差沿用原算法：相对于第一组恢复的板姿态；验证报告相对于验证集的平均旋转。两种旋转 RMS 的参考不同，不能直接当成同一种指标比较。验证也不对照已知的板绝对位置，整体偏差可能不体现在一致性指标里。

### 5. 发布静态 TF

确认 `--parent-frame` 是采集时的 TCP/法兰坐标系，`--child-frame` 是彩色 `camera_info` 中的光学坐标系：

```bash
python3 scripts/publish_handeye_tf.py \
  --result results/handeye_result.yaml \
  --parent-frame flange_link \
  --child-frame d405_color_optical_frame
```

上面的 child 名称只是示例，要替换成实际值；保持进程运行。程序从求解矩阵生成平移和四元数。现有相机驱动可能已经为光学坐标系发布 TF，应先设计清晰的单父节点 TF 树，避免对同一个 child 发布冲突变换。

如果仅在程序中转换视觉物体位姿，不必发布 TF，直接计算：

```python
T_base_object = T_base_tcp @ T_tcp_camera @ T_camera_object
```

本仓库不包含 FoundationPose、抓取规划、CAN 初始化或机械臂驱动。

## 目录

```text
scripts/              检测、采集、求解、验证、TF 发布及公共模块
config/board.yaml     实物标定板参数
examples/metadata/    可复现的原始位姿样本（求解 / 验证分开）
examples/reference/   原始最终标定和验证报告，仅作参考
tests/                合成数据、退化输入、真实数据回归检查
docs/PROVENANCE.md    来源、取舍与修改说明
```

## 上传和复用

将本目录作为独立 GitHub 仓库根目录上传即可；无需父工程、模型权重或原始备份。`.gitignore` 排除了新采集的 `data/`、运行生成的 `results/` 和 Python 缓存，但保留轻量示例。也可直接打包本目录分发，解压后按本文安装依赖。

源码来自个人工程，本次整理未替作者指定开源许可证。公开发布前可自行选择许可证并添加 `LICENSE`；没有许可证不等于授予他人任意使用、修改和分发的权利。
