# DexMimicGen 加 32×32 压电 Tactile Observation —— 方案 & 调研

> **历史文档 / 已过期:** 本文件是早期方案调研，里面的 `touch_grid`
> plugin 示例、tips-only 范围、TODO 状态已经不是当前实现。当前三种
> end effector 的最终位置记录见
> `tactile_recollect/TACTILE_LAYOUT_VALIDATION.md`。

> 目标:用 DexMimicGen 框架,给所有 demo 的 observation 加 32×32 压电式 tactile,
> 且 policy inference 时也把 tactile 作为 observation。

---

## 0. 这个仓库里有哪些灵巧手?

dexmimicgen 本身 **不含** 灵巧手模型,它只用字符串引用 **robosuite** 的 gripper/hand。
robosuite(`robosuite/models/grippers/`)里可用的手 / 夹爪:

### 灵巧手(多指)
| 注册名 | 类 | 文件 | 用途 |
|---|---|---|---|
| `InspireLeftHand` / `InspireRightHand` | InspireLeft/RightHand | `inspire_hands.py` | **Panda + dexterous hands** 任务用的 5 指手 |
| `FourierLeftHand` / `FourierRightHand` | FourierLeft/RightHand | `fourier_hands.py` | **GR-1 humanoid** 用的 5 指手 |
| `JacoThreeFingerDexterousGripper` | — | `jaco_three_finger_gripper.py` | 3 指 |
| `RobotiqThreeFingerDexterousGripper` | — | `robotiq_three_finger_gripper.py` | 3 指 |

### 平行夹爪 / 其它(非灵巧手)
`PandaGripper`(parallel-jaw,Threading/Transport/ThreePieceAssembly 用)、
`RethinkGripper`、`Robotiq85Gripper`、`Robotiq140Gripper`、`BDGripper`、
`XArm7Gripper`、`SuctionGripper`、`WipingGripper`、`NullGripper`。

### 任务 → embodiment(来自 environments.md)
- parallel grippers:TwoArmThreading / ThreePieceAssembly / Transport
- **Panda + dexterous hands(Inspire)**:TwoArmDrawerCleanup / BoxCleanup / LiftTray
- **Humanoid(GR-1)+ dexterous hands(Fourier)**:TwoArmCoffee / Pouring / CanSortRandom

### Inspire 右手 body 结构(贴 tactile site 的目标)
```
r_palm
r_thumb  → r_thumb_proximal_1 → r_thumb_proximal_2 → r_thumb_middle → r_thumb_distal   ← 指尖
r_index  → r_index_proximal  → r_index_distal                                          ← 指尖
r_middle → r_middle_proximal → r_middle_distal                                         ← 指尖
r_ring   → r_ring_proximal   → r_ring_distal                                           ← 指尖
r_pinky  → r_pinky_proximal  → r_pinky_distal                                          ← 指尖
```
左手同构,前缀 `l_`。Fourier 手指尖 body 形如 `R_thumb_distal_link` / `L_*`。
**5 个 `*_distal` body 就是放 32×32 tactile 的位置**(先指尖,跑通再扩 proximal/palm)。

→ 你真实机器是 Tesollo 灵巧手,**最贴近的是 Inspire(Panda + dexterous hands)**,
建议主攻 Inspire;parallel-jaw(PandaGripper 左右 `*_fingerpad`)用来最快跑通数据流。

---

## 1. 关键机制(已核实)

- DexMG env 继承 robosuite `TwoArmEnv`;demo 以 `(model=xml, states=flatten sim state)` 存储
  (见 `two_arm_dexmg_env.py: get_state()`)。
- replay / obs 提取:`reset_to(env,state)` → `env.edit_model_xml(model)` →
  `reset_from_xml_string` → 逐帧 `set_state`(见 `scripts/playback_datasets.py`)。
- **tactile 是接触状态的纯函数**,states 完全决定接触 → 可从已有 demo 的 states 重放出 tactile,
  **无需重新 teleop**。
- 注入点:override `edit_model_xml`(往存储 xml 注入 site + plugin sensor),
  或 playback 用 `--use_current_model` 用改过的当前模型。

## 2. 压电 → MuJoCo 映射

压电响应法向压力 → MuJoCo 法向接触力。
- 首选 `touch_grid` plugin:`nchannel=1`(法向 z)、`size=32 32` → 每指 32×32 图,对上真实 Tesollo。
- 稀疏刚性接触 → 图偏稀疏;若太空,升级 `touch_stress`(稠密高分辨,需被抓物体声明 SDF)。
- 想和真实 2-channel 一致就 `nchannel=2`(加第一切向),但物理意义未必对应。

---

## 3. 落地步骤

### P1 — tactile 注入器(给手指 distal body 加 site + plugin sensor)
```python
# dexmimicgen/utils/tactile_inject.py
import xml.etree.ElementTree as ET

INSPIRE_TIPS = {0: ["r_thumb_distal","r_index_distal","r_middle_distal",
                    "r_ring_distal","r_pinky_distal"],          # robot0 右手
                1: ["l_thumb_distal","l_index_distal","l_middle_distal",
                    "l_ring_distal","l_pinky_distal"]}          # robot1 左手 (按实际前缀调整)

def inject_tactile(xml_str, tips_by_robot=INSPIRE_TIPS, size=32, nchannel=1):
    root = ET.fromstring(xml_str)
    ext = root.find("extension")
    if ext is None: ext = ET.SubElement(root, "extension")
    ET.SubElement(ext, "plugin", {"plugin": "mujoco.sensor.touch_grid"})
    sensor = root.find("sensor")
    if sensor is None: sensor = ET.SubElement(root, "sensor")
    name2body = {b.get("name"): b for b in root.iter("body")}
    for rid, tips in tips_by_robot.items():
        for tip in tips:
            # gripper 前缀实际是 gripper{rid}_<tip>,按加载后的真实名字调整:
            body = name2body.get(tip) or name2body.get(f"gripper{rid}_{tip}")
            if body is None:  # 名字对不上就 grep 一次实际 model 再改
                continue
            sname = f"{body.get('name')}_touch"
            ET.SubElement(body, "site", {"name": sname, "type": "box",
                "pos": "0 0 0.004", "size": "0.006 0.006 0.002", "rgba": "1 0 0 0.2"})
            plg = ET.SubElement(sensor, "plugin", {"name": sname,
                "plugin": "mujoco.sensor.touch_grid", "objtype": "site", "objname": sname})
            for k, v in [("nchannel", str(nchannel)), ("size", f"{size} {size}"),
                         ("fov", "45 45"), ("gamma", "0")]:
                ET.SubElement(plg, "config", {"key": k, "value": v})
    return ET.tostring(root, encoding="unicode")
```
挂到 env(`two_arm_dexmg_env.py`):
```python
def edit_model_xml(self, xml_str):
    xml_str = super().edit_model_xml(xml_str)
    from dexmimicgen.utils.tactile_inject import inject_tactile
    return inject_tactile(xml_str)
```
> 坑:`touch_grid` 是 plugin sensor,必须有 `<extension>` 声明。先用原生 `<touch>`(单指标量)
> 验证数据流,再换 touch_grid。

### P2 — robosuite Observable 暴露 `robotX_tactile`
在对应 env 的 `_setup_observables()`:
```python
from robosuite.utils.observables import Observable, sensor
import numpy as np
H = W = 32

def _make_tactile_sensor(self, rid, tip_sites):
    @sensor(modality="tactile")
    def tactile(obs_cache, _names=tip_sites):
        out = []
        for nm in _names:
            sid = self.sim.model.sensor_name2id(nm)
            adr = self.sim.model.sensor_adr[sid]
            dim = self.sim.model.sensor_dim[sid]   # = nchannel*32*32
            out.append(self.sim.data.sensordata[adr:adr+dim].reshape(-1, H, W))
        return np.concatenate(out, axis=0)         # (n_tips*nchannel, 32, 32)
    return tactile

# 注册:observables[f"robot{rid}_tactile"] = Observable(...)
```

### P3 — 数据(两条路)
- **A 重新生成**:改完手模型 → 跑 mimicgen datagen → 新数据原生含 tactile(最干净)。
- **B retrofit 已有 HF 数据(推荐)**:写 `dataset_states_to_obs` 风格脚本,逐 demo 取
  `states`,用 tactile-augmented env `reset_to` 每帧 → `sim.forward()` → 读 sensordata →
  写回 `obs/robotX_tactile` 与 `next_obs/robotX_tactile`。robomimic dexmimicgen 分支自带
  `dataset_states_to_obs.py`,加一个 tactile key 即可。

### P4 — robomimic config + 训练
- `observation.modalities` 加 `robotX_tactile`:
  - 简单:当 `low_dim` flatten 成 `n_tips*nchannel*1024`;
  - 推荐:当图像模态,32×32 配小 CNN encoder(同 wm 工程 tactile ViT 思路)。
- 跑 BC-RNN(`scripts/generate_training_config.py` → robomimic `train.py`)。

### P5 — inference
不用额外改:robomimic 每步从 env 取 obs,只要 env 用同一 tactile-augmented 手模型、
config 列了该 key,policy 自动消费 tactile。

---

## 进度
- [x] P0 列出可用灵巧手 + 任务 embodiment 映射 + Inspire body 名
- [ ] P1 tactile 注入器(确认加载后真实 body 前缀:`gripperX_*` vs `rX_*`)
- [ ] P2 Observable 暴露 + obs dict 验证 (n_tips*nch, 32, 32)
- [ ] P3 数据:A 重生成 / B retrofit
- [ ] P4 robomimic config + encoder + 训练
- [ ] P5 inference 验证
