# Vid2ArtCAD 标注管线

按 `Vid2ArtCAD_Annotation_Pipeline.html` v0.2 实现的阶段化数据标注骨架。
目标是把资产和视频加工成带来源、字段级有效掩码、可复现的样本。

## 数据流

1. `S0 ingest`:登记原始文件、hash、许可和版本。
2. `S1/S2 fusion`:解析 Fusion Assembly 实例与关节,规范化为运动学资产。
3. `S3 geometry`:导出渲染几何;CAD 程序是可选层,缺失不阻断 G0。
4. `S4 motion`:生成 q(t)、相机、材质和遮挡计划。
5. `S5 render`:同一帧状态同步导出 RGB、深度、part ID。
6. `S6 labels`:派生 mask、框、轨迹、可见性和语言字段。
7. `S7 edit`:对有命名参数与锚点的 G2 资产生成编辑配对。
8. `S8 validate`:格式、几何、运动、可辨识性和字段来源检查。
9. `S9 package`:生成版本化 manifest 与设计族级 split。

## 快速开始

```bash
pip install -e .
vid2artcad init --root data
vid2artcad ingest --root data --source /path/to/fusion_assets
vid2artcad audit --data-root data/raw/batch_001 --output annoation/dataset_audit.json
OPENBLAS_NUM_THREADS=1 vid2artcad run --data-root data/raw/batch_001 --output-root annoation
vid2artcad package --output-root annoation
vid2artcad package-minimal --manifest-root annoation_final --cad-root a1.0.0_00 --destination data/published
vid2artcad validate --sample annoation/samples/<sample> 
vid2artcad edit --sample annoation/samples/<sample> --output annoation/edits/<id> --joint Rev1 --scale 1.5
vid2artcad video-import --video /path/to/video.mp4 --masks /path/to/uint16_masks --output annoation/real/<id>
vid2artcad video-import-multiview \
  --view cam01=/path/to/cam01.mp4 --view cam02=/path/to/cam02.mp4 \
  --mask-view cam01=/path/to/masks_cam01 --mask-view cam02=/path/to/masks_cam02 \
  --camera-view cam01=/path/to/camera_cam01.json --camera-view cam02=/path/to/camera_cam02.json \
  --output annoation/real/<id>
vid2artcad review --sample annoation/real/<id>
vid2artcad review --sample annoation/real/<id> --view cam02
vid2artcad propagate --sample annoation/real/<id> --view cam01 --seed-frame 120 --radius 30
vid2artcad decode-motion --reference annoation/samples/<reference> \
  --masks annoation/real/<id>/labels/part_id_cam01 \
  --camera annoation/samples/<reference>/camera_cam01.json \
  --output annoation/real/<id>/decoded_motion
vid2artcad sam2-propagate --sample annoation/real/<id> \
  --seed-frame 120 --model-id facebook/sam2.1-hiera-large
vid2artcad cotracker-track --sample annoation/real/<id> \
  --view cam01 --query-frame 120 --points-per-part 16
vid2artcad package --output-root annoation
vid2artcad queue --data-root data/raw/batch_001 --output-root annoation --hourly-rate-usd 25
vid2artcad queue-server --data-root data/raw/batch_001 --output-root annoation --port 8766
vid2artcad acceptance-audit --data-root data/raw/batch_001 --output-root annoation
vid2artcad expand-views --sample annoation/samples/<sample> --offset-degrees 90 270
vid2artcad status --root data
```

`ingest` 将原始资产复制到 `data/raw` 并写入 hash 清单；`audit` 统计资产和关节类型；
`run` 生成参考运动、双视角 RGB、深度、part ID、轨迹和质量报告；默认从 `--root/raw` 读取资产，也可用 `--data-root` 指定目录。当前 `run` 执行完整阶段链，不接受阶段子集。`package` 只把质量和标签均完整的样本写入 dataset manifest；split 以设计族为单位，同族变体不会跨集合，小于三个设计族时无法保证三个集合都有样本。
`package-minimal` 从已通过 manifest 的样本生成自包含交付集。每个样本只保留两个视频、`assembly.urdf`、`assembly.step`，以及 URDF `<mesh>` 实际引用的 `meshes/` 文件；缺少任何网格引用时拒绝打包。
`edit` 只改变指定关节的 q(t)，重新生成同步视频和标签；含闭环约束的样本会明确进入 review，等待闭环求解器处理。
`video-import` 接收真实 RGB 视频和外部模型生成的单通道 uint16 part mask/JSONL 轨迹；导入前会验证视频帧尺寸、mask 尺寸与帧数，不匹配时拒绝写入。没有 mask 时只登记视频并标记 `needs_review`，不会伪造真值。发布验收要求人工复核覆盖完整且至少有一个已命名的前景部件，空白掩码不能作为成功标注。
`video-import-multiview` 将同步的 `cam01/cam02/...` 视频、相机标定和 mask 作为一个样本导入；每个视角拥有独立的 `part_id`、review 状态和 checksum，帧数/fps 不一致会拒绝导入。
`review` 启动本机逐帧复核界面，支持播放/逐帧定位、部件 ID 画笔与擦除、从已有伪标签继续修改、标注员记录和复核进度；所有帧显式保存后才会标记 `human_review_complete`。`package` 会将完成复核的 `real/` 视频标注与合成运动学样本一起写入 manifest，并校验帧覆盖、uint16 标签和校验和。多个真实视频若来自同一设计族，可在导入时用 `--design-family-id` 绑定 split。

可选安装 `pip install -e '.[propagation]'` 启用 OpenCV Farneback 光流传播：先人工标注并保存种子帧，再设定帧范围传播。传播结果记录在 `review.json.pseudo_frames`，不计入人工复核进度；前后向光流不一致的像素写为 `65535`（待判定）。必须逐帧检查、修正并保存，才能进入数据集。

`decode-motion` 将人工确认或待复核的 `part_id` 质心与参考 CAD 的运动学树、相机标定联合拟合为 `decoded_motion/decoded_motion.npz`。默认只优化参考 `motion.npz` 中轨迹幅度大于阈值的关节；可重复传入 `--joint Rev1 --joint Rev2` 显式选定解码自由度。每帧用拟合雅可比的满列秩检查可观测性，`q_valid` 只标记被选中且可观测的坐标；未选关节不会被静态外观误判为有效。只有所有关节都被选择且各帧通过观测秩与残差门槛时，才发布完整 `motion`/`kinematics` 任务；否则状态为 `motion_decode_partial` 或 `review`，可作为待审结果但不会进入完整运动学数据集。支持重复 `--masks/--camera` 做多视角联合优化；参与融合的相机必须提供有效刚体外参、针孔内参、正帧率和无畸变图像标记，各视角帧数和帧率必须一致。多视角按相同帧索引配对，但真实采集时刻是否同步无法仅从这些文件独立验证，会在 `motion_decode_quality.json` 中明确记录为假设。

`sam2-propagate` 是可选的真实视频伪标签后端：先在 Review 中保存人工种子帧，再运行 SAM2 视频传播。它只写入 `pseudo_frames` 和 `sam2_propagation_runs`，不会扩大人工复核覆盖；安装 PyTorch 后还需按 Meta 官方 SAM2 仓库安装 `sam2` 包及模型权重。

`cotracker-track` 从人工种子 mask 为每个部件采样查询点，生成 `tracks_<view>.jsonl`；轨迹带有 `part_id`、可见性和置信度，但仍标记为 `pseudo`，不能跳过人工 mask 复核。

`queue` 会扫描原始 `data-root`、`real/` 和已导出的多视角样本，生成持久化 `annotation_queue.json`；原始资产会进入资格检查和标注生成待办，真实视频再按视角拆分复核、轨迹、运动解码，发布样本单独验收。任务记录领取人、开始/完成时间、耗时、失败原因与按小时费率估算的成本。`queue-server` 提供本地仪表盘，可领取任务并记录完成时间；标记完成前必须满足对应源产物门禁，刷新时若产物不再通过会重新打开任务。

`queue-server` 可将 `review/failed` 任务重新打开；`vid2artcad queue-reopen --queue <queue.json> --task-id <task-id> --notes <原因>` 提供同等 CLI 操作。每次重试保留历史 worker、结果、耗时和备注，成本统计累计全部尝试。

`acceptance-audit` 按 `docs/ACCEPTANCE.md` 逐项检查原始候选、已验证样本、设计族、模板证据和真实视频人工复核闭环；它只报告当前文件和验证器能证明的事实，缺少证据时保持 `pending`，不会把目标数量自动视为完成。

`expand-views` 在不重写已有 `cam01/cam02` RGB 的前提下，为合成样本增加 `cam03/cam04` 等视角，并重新导出对应的相机标定、part ID、深度、轨迹、质量和 checksums；视角数量由参数决定，旧样本仍保持可读取。

当前仓库提供可运行的 Fusion Assembly 资产适配、坐标/FK、复杂运动学树、闭环约束、
MuJoCo 回放、多视角渲染、标签导出、质量门控、增量回填和数据集打包。
SAM2/CoTracker3 等视觉伪标签适配器仍属于真实视频 S7 扩展层，不会覆盖 synthetic_gt 标签。

## 关键约定

- 长度 `m`,角度 `rad`,时间 `s`,世界坐标 z-up,相机 x-right/y-down/z-forward。
- `T_a_b` 将 b 系列向量映射到 a 系;矩阵按行存储。
- 参考装配状态定义为 `q=0`;`T_parent_joint * T_joint_child_ref == T_parent_child_ref`。
- `q` 是相对参考状态的位移/转角,不是源 JSON 中含义不明的 angle 字段。
- 限位保存 `lower/upper + lower_enabled/upper_enabled`,未知保存 null。
- `part_id` 背景 0,ignore 65535;整数标签不得有损压缩。
- 每个 field 的 `source` 只能是 `synthetic_gt / dataset_reference / measured / human_reference / pipeline_policy / pseudo / unknown`。
