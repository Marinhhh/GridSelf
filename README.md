# 社会等级认知实验系统

基于 PsychoPy 的行为实验平台，用于研究个体对社会等级结构的认知与重建能力。实验包含四个独立阶段：学习建表（A）、自我参照评分训练（B）、正式评分（C）和拖拽重建（D）。

## 📁 项目结构

```
项目根目录/
├── experiment/                    # 实验脚本目录
│   ├── phase_a.py                # 阶段A：学习/建表
│   ├── phase_b.py                # 阶段B：自我参照评分训练
│   ├── phase_c.py                # 阶段C：正式评分
│   ├── phase_d.py                # 阶段D：拖拽重建
│   ├── pools.py                  # 试次池管理器
│   └── config/
│       └── experiment.yaml       # 全局配置文件
├── Stimuli/                      # 刺激材料
│   ├── faces/                    # 他人面孔（P1.jpg ~ Pn.jpg）
│   └── self/                     # 自我面孔（PNG/JPG）
└── Data/                         # 数据输出目录
    ├── subjects/                 # 被试档案与个体化刺激表
    ├── A/                        # 阶段A输出
    ├── B/                        # 阶段B输出
    ├── C/                        # 阶段C输出
    ├── D/                        # 阶段D输出
    └── checkpoints/              # 断点续跑文件
```

## 🛠️ 环境准备

### 系统要求
- **操作系统**：Windows 10/11（推荐）
- **Python环境**：PsychoPy 2024.2.x
- **屏幕分辨率**：建议 1920×1080 或更高

### 依赖安装
实验脚本依赖以下 Python 包（通常 PsychoPy 已包含）：
- `psychopy`：实验呈现框架
- `numpy`：数值计算
- `pandas`：数据处理
- `matplotlib`：图表绘制
- `scipy`：科学计算
- `pyyaml`：配置文件解析

### 字体支持
系统已优先使用中文字体族（`SimHei` 等）。如出现字体警告，不影响实验正常运行。

## 🚀 快速开始

### 基本命令格式
在项目根目录打开 PowerShell，使用以下命令格式：

```powershell
# 示例：使用 PsychoPy 2024.2.4
F:/Psychopy2024.2.4/python.exe ./experiment/phase_a.py --subject 001
```

### 四个阶段运行命令

| 阶段 | 命令 | 说明 |
|------|------|------|
| **A** | `F:/Psychopy2024.2.4/python.exe ./experiment/phase_a.py --subject 001` | 学习建表，生成个体化刺激表 |
| **B** | `F:/Psychopy2024.2.4/python.exe ./experiment/phase_b.py --subject 001 --resume` | 评分训练，支持断点续跑 |
| **C** | `F:/Psychopy2024.2.4/python.exe ./experiment/phase_c.py --subject 001 --resume` | 正式评分，支持断点续跑 |
| **D** | `F:/Psychopy2024.2.4/python.exe ./experiment/phase_d.py --subject 001` | 拖拽重建 |

> **重要**：B/C/D 阶段依赖 A 阶段生成的个体化刺激表。常规流程必须先运行 A，再运行 B/C/D。
> 
> **应急选项**：`--allow-init` 参数可在找不到个体化表时临时生成，但不推荐常规使用。

## 📋 四个阶段操作指南

### 阶段 A：学习/建表

**目标**：生成被试个体化刺激表，为后续阶段提供基础数据。

**操作步骤**：
1. 运行脚本后，填写被试信息表单：
   - **SubjectID**：必须为字母/数字/下划线/横线组合
   - **其他信息**：按界面提示填写
2. 在自我图片下拉框中选择 `Stimuli/self/` 中的自我面孔文件
3. 按屏幕提示完成学习流程
4. 程序自动生成以下文件：
   - `Data/subjects/subj_<ID>_profile.json`：被试档案
   - `Data/subjects/subj_<ID>_stim_table.csv`：个体化刺激表

**输出文件说明**：
- `stim_table.csv` 包含每个刺激的 `com`（能力）、`pop`（受欢迎度）、`distance`（距离）、`angle_*`（角度）等字段

### 阶段 B：自我参照评分训练

**目标**：通过"距离任务+向量任务"进行训练评估，记录准确率并达到设定阈值。

**交互操作**：
- **距离滑杆（水平）**：鼠标拖动或 `←/→` / `A/D` 微调
- **能力滑杆（竖直）**：鼠标拖动或 `W/S` 控制
- **受欢迎度滑杆（竖直）**：鼠标拖动或 `↑/↓` 控制
- **确认评分**：按空格键确认当前三项评分

**实验流程**：
Self → 注视点 → P1评分 → 紫色注视点 → Self → 注视点 → P2评分

**数据输出**：
- `Data/B/subj_<ID>_trials.csv`：全量明细数据（含时序、正确性、反应时）
- `Data/B/subj_<ID>.csv`：精简版结果（核心指标）

**断点续跑**：支持 `--resume` 参数，使用 `Data/checkpoints/subj_<ID>_B.json` 文件

### 阶段 C：正式评分

**目标**：固定 block×trial 的正式评分，评分机制与数据格式与 B 阶段保持一致。

**交互操作**：与 B 阶段完全相同

**功能**：预留程序，后续可直接接入fNIRS或fMRI等设备，用于正式实验

**配置特点**：
- 时序参数可设置为定时（如每个评分 5 秒）
- 无早停机制，按固定试次数完成

**数据输出**：
- `Data/C/` 目录下的文件结构与 B 阶段一一对应，便于合并分析

**断点续跑**：支持 `--resume` 参数

### 阶段 D：拖拽重建

**目标**：呈现所有面孔（含自我）于左侧平铺区，右侧为 n×n 网格；参与者将面孔拖入网格，程序量化为实验坐标并评估正确性。

**界面布局**：
- **左侧**：所有面孔（包括自我）随机平铺排列
- **右侧**：n×n 网格区域，用于放置面孔

**交互操作**：
1. 按空格键退出指导语并开始任务
2. 鼠标按下移动头像，松开时若落在单元格内则**自动吸附**
3. 若单元格已被占用则退回原位
4. 当**全部放置**后，可按 **Enter** 确认
5. 否则到达**超时时间**会自动保存当前布局

**配置选项**：
- `can_drag_self`：是否允许拖动"自我"图片（false 则自我锁定在原位）

**数据输出**：

#### 主要数据文件：`Data/D/subj_<ID>_reconstruction.csv`

| 字段名 | 含义 | 说明 |
|--------|------|------|
| `StimID` | 刺激ID | 0=自我，1..N=他人面孔 |
| `IsSelf` | 是否自我 | 1=自我，0=他人 |
| `Placed` | 是否放置 | 1=已放入网格，0=未放置 |
| `CellIdx` | 单元格索引 | 吸附单元格的索引（从0开始） |
| `Row/Col` | 行列位置 | 单元格的行列坐标（从0开始） |
| `CenterX_pix/CenterY_pix` | 屏幕像素坐标 | 头像中心的屏幕坐标 |
| `ReconRawX/ReconRawY` | 连续实验坐标 | 从像素投影到实验坐标系的连续值 |
| `ReconX/ReconY` | 量化坐标 | 按步长量化后的坐标值 |
| `TrueX/TrueY` | 真值坐标 | 来自个体化刺激表的真实坐标（自我为0,0） |
| `AbsErrX/AbsErrY` | 绝对误差 | \|ReconX-TrueX\| 与 \|ReconY-TrueY\| |
| `IsCorrect` | 是否正确 | 在容差内同时满足X/Y为1，否则为0 |

#### 其他输出文件：
- `Data/D/subj_<ID>_reconstruction_summary.csv`：汇总统计（放置数量、正确率等）
- `Data/D/subj_<ID>_reconstruction.png`：当前布局截图

## ⚙️ 配置说明

配置文件位置：`config/experiment.yaml`

### 1. 量化范围（全局，A/B/C/D 共用）

```yaml
slider:
  component_ticks: [-9, -6, -3, 0, 3, 6, 9]   # 刻度端点，确定坐标范围
  component_step: 1.0                          # 量化步长（D阶段的ReconX/Y按此步长四舍五入）
```

### 2. Phase D 专用配置

#### 基本设置
```yaml
phase_d:
  timeout_s: 1800                # 拖拽总时长（秒）
  include_self: true             # 左侧是否包含自我
  can_drag_self: true            # 是否允许拖动自我（false则自我锁定）
```

#### 左侧平铺区
```yaml
  left_panel:
    width_frac: 0.35             # 左侧占屏宽比例（0-1）；越小→右侧网格可更大
    hpad_px: 24                  # 左右/上下留白，参与网格尺寸计算
    top_pad_px: 24               # 平铺顶端内边距
    vgap_px: 16                  # 头像竖向间距
```

#### 头像设置
```yaml
  tile:
    max_width_px: 200            # 头像最大边长；实际取min(此值, 单元格可用尺寸)
    border_px: 2                 # 边框厚度（仅保留占位）
    preload_thumb_px: 160        # 贴图预加载尺寸
```

#### 右侧网格
```yaml
  grid:
    n: 5                         # n×n单元数
    stroke_px: 3                 # 轴线/网格线粗细
    gap_px: 2                    # 单元内缩，防止头像贴边
    color: "white"               # 线条颜色
    show_cells: false            # 是否显示小格线（false=仅显示坐标轴）
    center_on_screen: false      # true=网格居中到屏幕；false=居右区域居中
    x_offset_px: 0               # 网格水平微调（+右移，-左移）
    y_offset_px: 0               # 网格垂直微调（+上移，-下移）
```

#### 坐标轴设置
```yaml
  axes:
    show: true                   # 是否绘制坐标轴
    range: [-9, 9]              # 坐标范围（省略则使用slider.component_ticks端点）
    stroke_px: null             # 轴线粗细（null=使用grid.stroke_px）
    color: null                 # 轴线颜色（null=使用grid.color）
```

#### 标签设置
```yaml
  labels:
    show: true                   # 是否绘制文本标签
    x: "能力"                    # 横轴标题
    y: "受欢迎度"                # 纵轴标题
    xmin: "-9"                   # 横轴左端文本
    xmax: "9"                    # 横轴右端文本
    ymin: "-9"                   # 纵轴下端文本
    ymax: "9"                    # 纵轴上端文本
    color: "white"               # 标签文字颜色
    offsets_px:                 # 标签偏移量
      x: 30                      # 横轴标题相对网格下边的偏移（+向下）
      y: 40                      # 纵轴标题相对网格左边的偏移（+向左）
      xmin: 30                   # 横轴左端文本偏移
      xmax: 30                   # 横轴右端文本偏移
      ymin: 40                   # 纵轴下端文本偏移
      ymax: 40                   # 纵轴上端文本偏移
```

### 3. Phase B/C 时序与阈值

#### Phase B 配置示例
```yaml
phase_b:
  self_s: 2.5                    # 自我呈现时长
  fixation_white_jitter_s: [0.5, 1.0]  # 注视点抖动时长
  p1_s: 5.0                      # P1评分时长
  purple_s: 0.5                  # 紫色注视点时长
  p2_self_s: 2.5                 # P2自我呈现时长
  p2_s: 5.0                      # P2评分时长
  blocks: 2                      # 组块数
  trials_per_block: 24           # 每块试次数
  criteria:
    accuracy_target: 0.95        # 准确率目标
    count_policy: "either"       # 计数策略
    trials_batch_size: 48        # 每轮追加试次数
```

#### Phase C 配置示例
```yaml
phase_c:
  self_s: 2.5                    # 自我呈现时长
  p1_s: 5.0                      # P1评分时长（定时）
  p2_s: 5.0                      # P2评分时长（定时）
  blocks: 2                      # 组块数
  trials_per_block: 24           # 每块试次数
```

### 配置调整效果说明

| 参数 | 调整效果 | 建议值 |
|------|----------|--------|
| `left_panel.width_frac` | 控制左右分栏比例 | 0.35（35%左侧，65%右侧） |
| `tile.max_width_px` | 头像大小上限 | 200px |
| `grid.n` | 网格密度 | 5×5（25个位置） |
| `grid.show_cells` | 是否显示小格线 | false（仅显示坐标轴） |
| `grid.center_on_screen` | 网格居中方式 | false（居右区域） |
| `axes.range` | 坐标范围 | [-9, 9] |
| `timeout_s` | 任务时长限制 | 1800秒（30分钟） |

> **重要**：修改配置文件后，需要重新运行脚本才能生效。

## 🎮 操作控件总览

### B/C 阶段滑杆操作

| 滑杆类型 | 鼠标操作 | 键盘操作 | 说明 |
|----------|----------|----------|------|
| **距离滑杆（水平）** | 拖动 | `←/→` 或 `A/D` | 控制距离评分 |
| **能力滑杆（竖直）** | 拖动 | `W/S` | 控制能力评分 |
| **受欢迎度滑杆（竖直）** | 拖动 | `↑/↓` | 控制受欢迎度评分 |
| **确认评分** | - | `空格键` | 确认当前三项评分 |

> **方向映射**：可在配置中设置"上加下减"或"上减下加"模式

### D 阶段拖拽操作

| 操作 | 说明 |
|------|------|
| **开始任务** | 按空格键退出指导语 |
| **拖拽头像** | 鼠标按下移动，松开时自动吸附到网格 |
| **冲突处理** | 单元格被占用时退回原位 |
| **完成确认** | 全部放好后按Enter键 |
| **超时保存** | 到达时间限制自动保存当前布局 |
| **退出程序** | 结束后按任意键退出 |

## 📊 数据输出字段详解

### Phase B/C 核心字段

#### 距离任务相关
| 字段名 | 含义 | 示例 |
|--------|------|------|
| `Resp_P*_distance` | 被试距离评分 | 2.5 |
| `TargetDist_P*` | 目标距离 | 3.0 |
| `AbsError_P*_distance` | 距离绝对误差 | 0.5 |
| `Correct_P*_distance` | 距离任务正确性 | 1 |
| `RT_*_distance_MS` | 距离任务反应时 | 1250 |

#### 向量任务相关
| 字段名 | 含义 | 示例 |
|--------|------|------|
| `Resp_P*_dCom` | 被试能力评分 | 1.5 |
| `Resp_P*_dPop` | 被试受欢迎度评分 | -2.0 |
| `Target_*_dCom/dPop` | 目标能力/受欢迎度 | 2.0/-1.5 |
| `AbsError_*_dCom/dPop` | 能力/受欢迎度绝对误差 | 0.5/0.5 |
| `Correct_*_vector` | 向量任务正确性 | 1 |
| `RT_*_vector_MS` | 向量任务反应时 | 2100 |

#### 综合指标
| 字段名 | 含义 | 说明 |
|--------|------|------|
| `Is_Correct_distance_trial` | 距离任务试次正确性 | 0/1 |
| `Is_Correct_vector_trial` | 向量任务试次正确性 | 0/1 |
| `Is_Correct_either_trial` | 任一任务正确性 | 0/1 |
| `AccuToDate_*` | 累计准确率 | 0.85 |

#### 时序信息
| 字段名 | 含义 | 说明 |
|--------|------|------|
| `Onset_*_MS/Offset_*_MS` | 刺激开始/结束时间 | 毫秒时间戳 |
| `FixWhite*_S` | 注视点呈现时长 | 秒 |
| `RT_*_first_MS/RT_*_final_MS` | 首次/最终反应时 | 毫秒 |

### Phase D 关键字段

详见上文"阶段 D：拖拽重建"部分的字段对照表。

## 🔧 常见问题与排查

### 1. 系统提示问题

| 提示信息 | 原因 | 解决方案 |
|----------|------|----------|
| "Monitor specification not found. Creating a temporary one…" | PsychoPy本地无Monitor定义 | 正常提示，程序自动创建临时设置 |
| "Helvetica Bold not found" | 系统缺少字体 | 正常提示，会回落中文字体族 |
| "no frames to write / getMovieFrame" | 截图缓冲问题 | 可忽略，确保窗口未被遮挡 |

### 2. 运行问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| **找不到个体化刺激表** | 未运行A阶段 | 先运行`phase_a.py`生成刺激表 |
| **窗口位置异常** | 显示器设置问题 | 检查`experiment.fullscreen`或`experiment.window_size` |
| **头像显示异常** | 图片文件缺失 | 检查`Stimuli/faces/`和`Stimuli/self/`目录 |
| **网格大小不合适** | 配置参数问题 | 调整`left_panel.width_frac`或`grid.n` |

### 3. 数据问题

| 问题 | 检查项目 | 解决方案 |
|------|----------|----------|
| **数据文件缺失** | 检查`Data/`目录 | 确保有写入权限 |
| **被试ID重复** | 检查SubjectID | 使用唯一ID，避免覆盖 |
| **断点续跑失败** | 检查checkpoint文件 | 删除损坏的checkpoint文件重新开始 |

## ⚠️ 安全与备份建议

### 数据安全
1. **唯一被试ID**：每个被试使用唯一的`SubjectID`，与数据文件同名，避免覆盖
2. **及时备份**：实验完成后及时备份`Data/`目录
3. **权限检查**：确保程序对`Data/`目录有写入权限

### 实验前检查清单
- [ ] 检查图片是否齐全（`Stimuli/faces P1..Pn`）
- [ ] 确认自我图片在`Stimuli/self/`目录中
- [ ] 验证配置文件`experiment.yaml`设置正确
- [ ] 测试屏幕分辨率和窗口设置
- [ ] 确认被试ID唯一性

### 数据管理
- **定期备份**：建议每次实验后立即备份数据
- **版本控制**：重要配置修改前备份原文件
- **文件命名**：严格按照`subj_<ID>_*`格式命名

## 📞 技术支持

如遇到技术问题，请检查：
1. PsychoPy版本是否为2024.2.x
2. 所有依赖包是否正确安装
3. 配置文件格式是否正确
4. 刺激材料是否完整

---

**版本信息**：基于 PsychoPy 2024.2.x 开发  
**最后更新**：2025年10月  
**适用系统**：Windows 10/11
