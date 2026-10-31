# 实施设计

## 目录

```text
pipeline/
  core/           schema、任务状态、运动计划
  fusion/         Fusion Assembly 解析边界与校验
  kinematics/     统一坐标、FK、关节扰动重放
  render/         渲染后端接口、标签投影
  validation/     三层检查与质量分流
  cli/            阶段化入口
data/
  raw/            只读原始文件
  staging/        资产与片段中间产物
  published/      manifest 与发布版本
```

## 阶段契约

每个阶段读取清单和上游文件,输出新版本文件与结构化报告。阶段按 S0-S9 命名:
ingest、instances、kinematics、geometry、motion、render、labels、edit、validate、package。
任务键为 `source_sha256 + stage + config_hash + seed`。

## 接收门槛

试运行使用文档建议起点:

- 刚体误差 <= 1e-5。
- 参考位姿平移 <= 1e-4 * D,旋转 <= 1e-4 rad。
- 非边界投影中位数 <= 1 px。
- 重建 95 分位双向表面距离 / D <= 0.01。

数值须在人工校准集上调整后才可固定为发布门槛。

## 不做的事

- 不把未解析的源 angle 直接写成 canonical q。
- 不把 enabled=false 的 0 限位当作零行程。
- 不把伪标签改写成真值。
- 不用有损视频保存整数 part ID。
- 不通过删除闭环边制造独立关节。

## 复杂运动学图

`pipeline.kinematics.graph.normalize_graph` 是源装配图与渲染导出之间的规范化边界：

- `RigidJointType` 先合并为刚体组件，避免把固定连接误当成可动自由度；
- 非固定关节保留为图边；
- 每个连通分量选择确定性的 spanning forest 供 URDF/MuJoCo 树结构使用；
- 未被选入 forest 的边必须保存在 `loop_constraints.json`，不得丢弃；
- `roots` 支持多个独立装配分量；
- `dof_order` 明确记录多自由度关节的每个坐标，不能再假设一个 joint 对应一个 q 列。

URDF 只承担树的序列化，不是完整运动学真值。完整真值应同时保存：

```text
canonical graph + tree edges + loop edges + dof_order + q(t) + T_world_link(t)
```

闭环轨迹生成应使用树边做前向运动学，并以 loop edges 的位姿/轴线残差作为约束求解；求解失败或残差超阈值时进入复核，不得发布样本。

Fusion 关节的 canonical 化规则位于 `pipeline.fusion.adapter.source_joint_dof_spec`：

| 源类型 | canonical 类型 | DOF 顺序 |
| --- | --- | --- |
| `RigidJointType` | `fixed` | 无 |
| `RevoluteJointType` | `revolute` | rotation(primary) |
| `SliderJointType` | `prismatic` | translation(primary) |
| `CylindricalJointType` | `cylindrical` | rotation(primary), translation(primary) |
| `PinSlotJointType` | `pin_slot` | rotation(primary), translation(secondary) |
| `BallJointType` | `ball` | rotation vector(primary, secondary, tertiary) |
| `PlanarJointType` | `planar` | translation(secondary), translation(tertiary), rotation(primary) |

每个 DOF 的限位必须从对应的源字段单独保留；禁用的源限位不能转换成零行程。源关节原文同时写入 provenance，便于发现 Fusion 版本字段变化。

闭环求解统一使用 `pipeline.kinematics.solver.KinematicSolver`：

```text
q[dof_order]
  -> tree joints: T_parent * T_parent_joint * M(q) * T_joint_child_ref
  -> loop joints: pose residual (translation + rotation-vector)
  -> least_squares / prescribed driver DOFs
  -> q, T_world_link, closure residual
```

发布前必须检查 `SolveResult.success` 和残差阈值。`motion.npz` 应保存完整的 `dof_order`，而不是只保存树关节名称；`T_world_link` 必须来自同一次求解，不能由 renderer 重新独立计算。渲染器通过 `joint_runtime.json` 将 canonical DOF 映射到 MuJoCo 的串联 scalar joints 或原生 ball joint。
