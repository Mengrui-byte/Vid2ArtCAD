# 固化标注管线

## 正式运行

从仓库根目录执行：

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -m pipeline.render.auto_annotation \
  --data-root data/raw/batch_001 \
  --output-root output/annoation
```

默认行为：

- 已通过质量门控的样本直接复用；
- 每个资产最多激活两个 revolute/slider 关节；
- 先写入 `.building` 临时目录，完整成功后才发布到 `samples/`；
- 单个资产失败不会中止其他资产；
- `auto_annotation_status.json` 保存本次每个资产的状态和错误；
- 渲染后端不可用时只报告 `blocked_render_backend`，不会生成半成品。
- 已存在的参考目录或 `.building` 暂存目录不会被自动删除；遇到冲突会停止该资产并保留原目录。

## 预览与冲突处理

预览使用独立命令，不会写正式 `references/`、样本清单或正式运行状态：

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -m pipeline.render.auto_annotation \
  --data-root data/raw/batch_001 \
  --output-root output/annoation \
  --asset <asset-folder> \
  --preview
```

预览文件位于 `previews/references/<asset-id>/`，结果位于 `previews/preview_status.json`。验收预览后，再用正式命令生成参考与样本。正式输出目录若已有不完整或来源不明的参考/暂存目录，不要删除或改名覆盖；先核对来源，并在经批准的版本化输出位置重跑。

正式生成前必须记录人工审核。检查预览目录中的 `contact_sheet.png`、运动轨迹和质量报告后执行：

```bash
vid2artcad preview-review --data-root data/raw/batch_001 \
  --output-root output/annoation --asset <asset-folder> \
  --decision approved --reviewer <name> --notes "运动与结构已核对"
```

有错误时将 `--decision` 设为 `rejected`，并在备注中说明原因。审核以源装配、预览图片、运动、质量报告及相关管线代码的 hash 绑定；这些证据变化后必须重新预览和审核。`queue` 会把已生成预览的资产列为 `preview_review` 任务。历史已完成样本继续复用，不因该新增门禁而回溯阻断。

## 多人任务队列

任务领取、完成和刷新通过队列旁的锁文件串行化，避免并发 worker 重复领取或刷新覆盖领取状态。完成操作必须满足对应的源产物门禁；刷新时若已完成任务的源产物回退，任务会重新打开。`review` 和 `failed` 状态可从仪表盘或 `queue-reopen` 命令显式重开；每次尝试的 worker、结果、耗时与备注都会保留并累计到成本统计。

同一样本的不同视角 reviewer 共用样本级锁，写入 part 名称映射、mask、审核状态和 checksums 时不会互相覆盖。视频导入时会为每个视角登记预期标签目录；无初始 mask 的目录由首次人工保存创建。

Review 界面在切帧、播放或关闭页面时会提示未保存的画布修改；保存和帧掩码加载期间会暂时禁止画布输入，避免异步请求覆盖正在进行的编辑。传播只接受已保存的人工种子帧。

需要批处理在有失败时返回非零退出码时，加上 `--strict`：

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -m pipeline.render.auto_annotation \
  --data-root data/raw/batch_001 \
  --output-root output/annoation \
  --strict
```

## 质量门控

正式样本必须满足：

- `quality.json.status == "multiview_replay_passed"`；
- 两个视频均为完整 360 帧、30 FPS；
- MuJoCo FK 位姿和旋转误差通过检查；
- 两视角方位分离为 180 度；
- `cam01` 与批准 reference 字节一致；
- 输入 source 未被修改。

真实视频样本还必须为每个出现于 mask 的非背景 part ID 提供唯一、非空名称，映射保存在 `human_part_mapping.json`；缺帧编号、未命名 ID 或伪标签未人工复核都会阻止发布。

## 规范运动学产物

导入器先把 Fusion occurrence/joint 图转换为 `KinematicSolver`：

- 固定、旋转、滑动、圆柱、销槽、球和平面关节有明确的 DOF 顺序；
- 从源图选择确定性的 spanning tree，非树边写入 `loop_constraints.json`；
- `motion.npz` 同时保存渲染用 `q`、完整模型用 `q_all` 和 `all_joint_order`；
- `kinematic_model.json` 保存树边、闭环边、根和逆向遍历标记。
- `joint_runtime.json` 保存 canonical q 到 MuJoCo scalar/ball qpos 的显式映射；多自由度关节不再被压缩成一个单轴关节。
- 源图断连时，每个连通分量以独立 world mount 保持导入 occurrence 姿态；不推断分量间关节，并在 `kinematic_model.json` 与 `quality.json` 明确记录该策略。

几何缺失、断连、参考模型不一致或闭环无法满足的资产进入 `review_queue.json`，不会被降级成单自由度或伪造通过。

只有满足这些条件的目录才会被增量运行识别为 `already_complete`。

## 环境预检

运行前会创建一个 16x16 的 MuJoCo renderer 进行预检。macOS 无图形上下文时，状态会记录为：

```text
blocked_render_backend
```

这时应在有效桌面会话中运行，或配置项目支持的离屏渲染后端；不要把失败目录手动改名为正式样本。

## 资产导入原则

源资产必须通过 `Assembly` 的几何一致性、关节轴一致性、连通性和 FK 检查。缺失 body、嵌套 occurrence、闭环或特殊 joint 类型不应静默降级；应新增专用 adapter 或进入人工复核队列。
