# NIO 自动驾驶数据 → Rerun 可视化转换器

## 项目概述

本项目实现将蔚来（NIO）SIRIUS（NT2）平台自动驾驶车辆采集的原始数据转换为 [Rerun](https://github.com/rerun-io/rerun) 可视化工具支持的 `.rrd` 格式。

**核心功能：**
- 自动解析 NIO 车端采集的原始数据包（`.zip` 格式）
- 支持摄像头视频（H.264/H.265）、激光雷达点云、感知目标的转换
- 输出标准 Rerun `.rrd` 文件，可用 `rerun` 命令直接查看


**开发配置:**
#### install uv for project manangement
`brew install uv ffmpeg`

#### install all the python deps for project
`uv sync`

#### run the format transformer
`uv run nio_to_rerun.py data.zip output.rrd --max-frames 20`

#### run the rerun viewer
`uv run rerun output.rrd`
