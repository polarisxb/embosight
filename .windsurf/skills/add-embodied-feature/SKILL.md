---
name: add-embodied-feature
description: 为 EmboSight 添加具身功能（手臂控制、抓取、导航）的开发指南。当涉及 move_arm_to、grasp、机械臂、IK、动作执行时自动调用。
---

## 具身功能开发指南

### 当前状态

- `env_wrapper.py` 中 `move_arm_to()` 目前是 no-op
- pipeline 只有感知闭环（observe → plan → observe），缺少行动闭环
- 需要实现：手臂主动感知 + 目标抓取

### RoboCasa/Robosuite 动作空间

PandaMobile (PandaOmron) 使用复合控制器 (`default_pandaomron.json`)：
- **手臂**: OSC_POSE — 3D 位置增量 + 3D 旋转增量 = 6D
- **夹爪**: 1D (-1 打开, +1 关闭)
- **底盘**: 2-3D (x, y, θ)

关键观测数据：
```python
obs['robot0_eef_pos']     # 末端执行器位置 (3D)
obs['robot0_eef_quat']    # 末端执行器四元数 (4D)
obs['robot0_gripper_qpos'] # 夹爪关节位置
```

### 手臂控制实现要点

```python
def move_arm_to(self, target_pos, max_steps=200, threshold=0.02):
    import numpy as np
    target = np.array(target_pos[:3])
    
    for step in range(max_steps):
        current = self._latest_obs['robot0_eef_pos']
        delta = target - current
        if np.linalg.norm(delta) < threshold:
            return True
        
        # 归一化 + 限幅
        direction = delta / max(np.linalg.norm(delta), 1e-6)
        action = np.zeros(self._env.action_dim)
        action[0:3] = direction * min(np.linalg.norm(delta), 0.05)
        
        obs, _, done, _ = self._env.step(action)
        self._latest_obs = obs
    return False
```

### 抓取流程

1. **开夹爪** → 设置 gripper action = -1
2. **移到预抓取位置** → 目标上方 10-15cm
3. **下降到抓取位置** → 接近物体
4. **关夹爪** → 设置 gripper action = +1
5. **提升** → 提起物体

### Pipeline 行动执行步骤

在 `pipeline.py` 的 Step 4 (聚合) 之后加 Step 5：
1. LLM 判断是否需要执行动作（查询含"拿""取""递"等关键词）
2. 从 StructuredDescription.positions 中提取目标位置
3. 调用 `env.move_arm_to()` + `env.grasp_at()`
4. 再拍一张确认照片

### 注意事项

- robosuite `env.step()` 返回 `(obs, reward, done, info)`
- `done=True` 时 episode 结束，需要处理
- 动作值通常归一化到 [-1, 1]
- eye-in-hand 摄像头跟随手臂移动，适合做主动感知闭环
- 每次 `env.step()` 后 `self._latest_obs` 必须更新
