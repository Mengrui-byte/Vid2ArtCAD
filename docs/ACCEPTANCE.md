# 试运行验收

1. P0:至少 3 个不同设计族的已接收机构通过独立 MuJoCo 重放。审计逐帧比较 FK 位姿，要求样本包含 q=0 和活动关节正负扰动，并校验相机外参、焦距/FOV 与主点的一致性；任一项失败不计数。
2. P1:20 个 Fusion 候选产生接收/拒绝原因统计,每个接收资产 4 段视频。
3. P2:累计 100 个接收资产,按设计族划分,发布 manifest 和成本统计。
4. P3:3 类原生模板生成 G0/G1/G2 分层样本,编辑后关节与锚点同步更新。样本 `sample.json` 的 `native_template` 必须声明 `template_id` 和样本内 manifest 路径；manifest 声明 `source_format` (`f3d`/`f3z`) 及相对 `data_root` 的 `native_source.path` 和 SHA-256。每个模板需覆盖全部三个 generation_tier。至少一个通过样本验证的编辑产物须在 `edit.json` 中引用该模板样本，并提供 `joint_sync`、`anchor_sync`（`status=passed`、`max_error_m <= 1e-6`）及存在于编辑目录内的 `synchronized_outputs`。
5. P4:20 段固定机位真实视频完成伪标签与人工复核闭环。

每个发布样本必须有 provenance、quality、任务可用性列表;未知字段保持 null/unknown。

审计分别报告结构验收样本数与独立运行时重放通过/失败样本数。P1/P2 按结构验收状态计数；P0 仅按独立重放通过的不同设计族计数，不能用结构验收数量替代。

P4 对多视角真实视频要求每个视角都同时具备伪标签来源证据和完整人工复核。导入伪标签时保留 `labels.origin`；审核完成后 `labels.source` 转为 `human_reference`，`origin` 不变。传播生成的伪标签由各视角 `review.json.pseudo_frames` 留证。
