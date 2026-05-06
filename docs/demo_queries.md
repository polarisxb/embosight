# 比赛演示 Query 集

## A: 纯感知 (action_type=none)
1. 桌子上有什么？
2. 我的药瓶在哪里？
3. 周围有什么危险吗？

## B: 抓取 (action_type=grasp)
4. 帮我拿药瓶
5. 给我一个杯子
6. 帮我把刀拿开

## C: 复合任务
7. 我想喝水，帮帮我
8. 我要吃药，需要药瓶和水

## 演示流程
1. 启动 viewer: `python scripts/run_demo.py --query "桌子上有什么" --visualize`
2. 跑 A 类 query：展示主动观察 + 五维度描述
3. 跑 B 类：展示风险感知运动 + 抓取 + 验证
4. 现场让裁判提一个新 query：证明不是预录制

## 命令速查
```bash
# 纯感知
python scripts/run_demo.py --query "桌子上有什么" --visualize

# 抓取
python scripts/run_demo.py --query "帮我拿药瓶" --visualize

# 端到端测试
python scripts/test_embodied.py --query "帮我拿药瓶" --visualize
```
