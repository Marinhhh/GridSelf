\# 全局规则｜Global Rules for Cursor



\## 0) 角色与目标



\* 你是本项目（PsychoPy 行为实验）的\*\*唯一开发者\*\*。

\* 目标：实现并维护 \*\*4 个脚本 + 1 个配置\*\* 的工程，快速修复问题、最小改动、严格遵守数据与流程契约。



---



\## 1) 允许的文件与目录（禁止越界）



\* 仅允许存在/修改：



&nbsp; \* `phase\_a.py`, `phase\_b.py`, `phase\_c.py`, `phase\_d.py`

&nbsp; \* `config/experiment.yaml`（单一真源）

&nbsp; \* 目录：`Stimuli/faces/`, `Stimuli/self/`, `Data/A|B|C|D/`, `Data/subjects/`, `Data/checkpoints/`

\* \*\*禁止\*\*创建任何其它文件或目录（如 `main.py`、`utils/`、`demo/`、`tests/`、`README` 等），除非指令明确允许。



---



\## 2) 关键不变量（不可破坏）



\* \*\*S=0 自我面孔\*\*：来自 `Stimuli/self/`，坐标固定 `(com=0, pop=0)`，可在 \*\*A/B/C\*\* P1/P2 出现；在 \*\*D\*\* 固定显示，不可拖拽。

\* \*\*他人面孔 1..80\*\*：文件名 `P1.png … P80.png`，坐标为整数格点 `\[-9, +9]`（不含原点），每被试固定一次后跨阶段复用。

\* \*\*一行=一个 trial（A/B/C）\*\*：每个 trial 只写\*\*一行 CSV\*\*；B/C 的 P1/P2 评分写在同一行的不同列；trial 未完成不得写半行。

\* \*\*四阶段时序与阈值\*\*：



&nbsp; \* A：学习/测试（同时呈现）+ 交错（序列），\*\*直到总体 ≥90%\*\*；

&nbsp; \* B：评分训练，\*\*直到总体 ≥95%\*\*（以评分次数为分母）；

&nbsp; \* C：评分正式 \*\*3×48\*\*，无早停；

&nbsp; \* D：二维拖拽重建（S 锁定在原点），导出明细与汇总。

\* \*\*自我试次配额\*\*：A/B/C 各 block 额外注入含 S 的 trial（数量由 YAML 控制），\*\*不计入他人角度桶均衡\*\*统计。

\* \*\*角度桶均衡\*\*：仅对他人（1..80）按 12 桶均衡。

\* \*\*参数唯一真源\*\*：所有参数、颜色、时序、阈值、配额、路径、抖动均从 `config/experiment.yaml` 读取，\*\*不得硬编码\*\*。



---



\## 3) 配置发现与首运行向导



\* 启动任意 `phase\_\*.py`：



&nbsp; 1. 尝试加载 `--config`; 否则默认 `config/experiment.yaml`；

&nbsp; 2. 若不存在，弹「首运行向导」创建\*\*默认模板\*\*到 `config/experiment.yaml`，再继续；

&nbsp; 3. 读取后计算 `ConfigChecksum=SHA256(contents)` 与 `SchemaVer=meta.schema\_version`。

\* 在\*\*每条输出行\*\*写入：`SchemaVer`、`ConfigChecksum`。



---



\## 4) 被试信息与自我图绑定



\* 启动弹 `gui.DlgFromDict`：`SubjectID, Name, Sex, Age, Handedness, Notes, Phase(自动), SessionID(自动), SelfImageFile(从 Stimuli/self 选择)`;

\* 写入/更新 `Data/subjects/subj\_<ID>\_profile.json`，包含 `SelfImageFile` 与 `SelfImageSHA256`、`NameHash`（SHA256）。

\* \*\*CSV 与结果中不写明文 Name\*\*，仅写 `NameHash` 与 `SelfImageSHA256`。



---



\## 5) 被试×刺激坐标表（仅他人）



\* 路径：`Data/subjects/subj\_<ID>\_stim\_table.csv`；A 首次生成、B/C/D 只读。

\* 字段：`SubjID,StimID,com,pop,angle\_deg,angle\_bin,dist,seed,created\_at,schema\_version,checksum`。

\* \*\*S=0 不在此表\*\*。每次读取需校验 `checksum`，不一致则报错退出。



---



\## 6) CSV 架构（列名契约）



所有写盘前\*\*严格校验列顺序\*\*，首次写表头，之后写入必须完全一致；trial 未完成\*\*不写行\*\*。



\### A：`Data/A/subj\_<ID>\_trials.csv`



```

SubID,SessionID,Phase,Day,Block,Trial,NameHash,SchemaVer,ConfigChecksum,SubjectMapChecksum,SelfImageSHA256,Seed,

Type,CueDim,

P1,P2,

P1\_com,P1\_pop,P2\_com,P2\_pop,

P1\_angle\_deg,P2\_angle\_deg,P1\_angle\_bin,P2\_angle\_bin,

P1\_dist,P2\_dist,

C1,C2,DeltaDim,

Choice,Correct\_Choice,

Onset\_Pair\_MS,Offset\_Pair\_MS,

Onset\_P1\_MS,Offset\_P1\_MS,Onset\_P2\_MS,Offset\_P2\_MS,

RT\_Choice\_MS,Timeout,

FixWhite\_S,FixPurple\_S,FixGreen\_S,ITI\_S

```



\### B / C（相同表头）：`Data/B|C/subj\_<ID>\_trials.csv`



```

SubID,SessionID,Phase,Day,Block,Trial,NameHash,SchemaVer,ConfigChecksum,SubjectMapChecksum,SelfImageSHA256,Seed,

P1,P2,

P1\_com,P1\_pop,P2\_com,P2\_pop,

P1\_angle\_deg,P2\_angle\_deg,P1\_angle\_bin,P2\_angle\_bin,

P1\_dist,P2\_dist,

Onset\_P1\_MS,Offset\_P1\_MS,RT\_P1\_first\_MS,RT\_P1\_final\_MS,Resp\_P1,Timeout\_P1,TargetDist\_P1,AbsError\_P1,Correct\_P1,

Onset\_P2\_MS,Offset\_P2\_MS,RT\_P2\_first\_MS,RT\_P2\_final\_MS,Resp\_P2,Timeout\_P2,TargetDist\_P2,AbsError\_P2,Correct\_P2,

FixWhite\_S,FixGreen\_S,ITI\_S,

Correct\_Overall,AccuToDate

```



\### D 明细/汇总（独立文件）



\* 明细：`Data/D/subj\_<ID>\_reconstruction.csv`



```

SubID,SessionID,NameHash,SchemaVer,ConfigChecksum,SubjectMapChecksum,SelfImageSHA256,

StimID,TrueX,TrueY,ReconX,ReconY,ReconRawX,ReconRawY,

Snap,DragCount,FirstDragMS,LastDragEndMS,DistError,AngleErrorDeg

```



\* 汇总：`Data/D/subj\_<ID>\_reconstruction\_summary.csv`



```

SubID,SessionID,NameHash,SchemaVer,ConfigChecksum,SubjectMapChecksum,SelfImageSHA256,

N,MeanDistErr,MedianDistErr,SpearmanDistCorr,ProcrustesD,

TotalDragCount,TotalTimeMS,ScreenshotPath

```



> \*\*S=0 写法\*\*：当 `P?=0` → `P?\_com=0,P?\_pop=0,P?\_dist=0`，`P?\_angle\_\*` 置空；`TargetDist\_P?=0`；渲染自 `SelfImageFile`。



---



\## 7) 采样与阈值



\* \*\*A\*\*：学习差 1 级、测试可跨级、交错按定时序列；含 S 配额；总体 ≥\*\*0.90\*\* 才结束，否则按 `trials\_batch\_size` 追加。

\* \*\*B\*\*：含 S 配额；以\*\*评分次数\*\*为分母（P1+P2）累计 ≥\*\*0.95\*\*；不足则按 `trials\_batch\_size` 追加。

\* \*\*C\*\*：固定 \*\*3×48\*\*；他人角度均衡；S 配额不计入均衡。

\* \*\*D\*\*：S 固定在 (0,0)；计算 Spearman/Procrustes/误差并导出截图。



---



\## 8) 运行/恢复/可复现



\* 支持 `--config/--subject/--resume/--allow-init`；`--resume` 从\*\*最后完整 trial\*\*继续。

\* 随机数：`experiment.seed` + `SubjectID` 派生子种子，所有生成可复现。

\* 每个 trial 更新 `Data/checkpoints/subj\_<ID>\_<PHASE>.json`（记录 Block/Trial/RNG/完成索引）。



---



\## 9) 错误处理与日志



\* GUI 取消或必填缺失（如 SelfImageFile）：\*\*安全退出\*\*，不写数据。

\* 校验失败（表头不匹配、checksum 不一致）：\*\*立即报错并退出\*\*。

\* 日志打印\*\*精简关键信息\*\*：阶段/Block/Trial、累计准确率、角度桶计数、文件路径；避免冗长叙述。



---



\## 10) 性能与可维护性



\* \*\*最小必要修改\*\*：只改与问题直接相关的代码；\*\*不要\*\*大范围重构或重命名。

\* \*\*禁止\*\*占位/临时代码、禁用测试、注释掉关键校验；\*\*不得\*\*引入新依赖。

\* 依赖仅限：`psychopy`, `numpy`, `pandas`, `pyyaml`, `scipy`。



---



\## 11) 回复格式（固定模板，确保高效）



\*\*你的每次回复必须严格使用以下结构，避免闲聊与复述：\*\*



1\. `SUMMARY`（≤5行）：概述问题与修复点（不重复大段上下文）。

2\. `PATCHES`：对\*\*受影响文件\*\*给出\*\*最小 diff\*\* 或 \*\*替换片段\*\*（使用清晰代码块；若是插入/替换，请包含可搜索锚点注释）。

3\. `MIGRATION`：是否需要迁移/清理旧数据或补写表头；提供 1-2 步骤命令/说明。

4\. `CHECKS`：提交前的自检清单（包含列校验通过、S=0 正确写入、达标逻辑正确、恢复点可继续等）。

5\. `NEXT`：若仍存在潜在问题或可选优化，列出\*\*最多 3 条\*\*下一步建议（可选）。



> \*\*不要\*\*重复粘贴未修改部分；\*\*不要\*\*生成未授权的新文件；\*\*不要\*\*更改既定列名/表头；\*\*不要\*\*输出冗长解释。



---



\## 12) 遇到不明确场景时



\* 若存在\*\*阻塞性歧义\*\*（例如：确需新增列或改目录才能完成功能），\*\*先提出 1–3 条具体选项\*\*（成本/影响/风险一句话），\*\*默认选成本最低\*\*方案并继续实现；避免来回追问。

\* 非阻塞性细节（变量名、注释风格）直接按现行风格处理，\*\*不要停下来询问\*\*。



---



\## 13) 质量门槛（验收即通过）



\* 打开任意 `phase\_\*.py` 即可运行；缺 `config` 自动创建模板；GUI 正常；写入 trial 单行、列校验通过。

\* `Data/subjects/subj\_<ID>\_stim\_table.csv` 仅含他人 1..80；S=0 不在表中；跨阶段 checksum 一致。

\* 当 `P?=0`：列值按约定写入，渲染自 SelfImageFile；B/C 正确判定与阈值统计包含 S 评分。

\* A ≥90%、B ≥95%、C=3×48、D 指标导出；恢复点可无损继续。



