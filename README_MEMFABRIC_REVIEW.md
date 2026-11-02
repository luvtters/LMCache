# MemFabric worker 路线的原生 L1 core 配套

配套 Ascend 为 `LMCache-Ascend` 的同名目标分支。当前提供原生 L1 manager 的显式 memory manager 注入入口；MemFabric HOST 映射与搬运实现在 Ascend 仓库。默认未注入时沿用原生分配路径。显式注入的外部 arena 仍有写暂存或活跃读保护时拒绝关闭，避免关闭过程先释放正在使用的对象。

**当前仅验收 native L1 管理与 BM 真实数据搬运基础 gate；完整 vLLM connector、模型推理、P/D 和性能尚未完成。** L2 兼容路线不依赖本分支。

## 固定源码与保留范围

- 官方基线：`6913259264dbebb2fa5ba1665606c40ffc5c6661`。
- 本目标起点：core `26e0d7e01f95d81edcc3c8b7d77c89399c851a23`，配套 Ascend `13ad6e39e6a1b300dc310a7219d1d6bcfae97430`，同时应用本目标两仓库差异。
- 开发分支：`work/memfabric-l1-compat-20261102`；长期检视分支：`review/memfabric-worker-direct`。是否完成提交及同步，以实际 tip 和本轮记录为准。
- 旧 Direct 专用 scheduler adapter 工厂和 lookup state hook 已恢复官方实现，其专用文档与测试同时移除。
- 保留通用 transfer channel 按需导入及 native callback 关闭修复；未随旧 Direct 删除。
- 旧 worker 直传只归档于本地 tag `archive/memfabric-worker-direct-pre-l1-20261102`，不与新原生 L1 并行维护两套对象管理。

## 验证

2026-11-02，M8 对同名独立目标分支及本轮源码差异完成 native L1 CPU gate（退出码 0）和 18 MiB 真实 NPU→BM HOST 原生 L1→NPU gate（pool、worker 均退出码 0）。这些不是完整模型或 P/D 验收。随后清理旧 Direct 和专用 hooks，已验证的 core 注入、BM、L1 backing 与 gate 方法体保持不变；清理后的定向 CPU 回归由本轮记录单列。

直接复用配套 Ascend 的 CPU 入口，无需另建测试框架或复制实现：

```sh
python /absolute/path/to/worker-ascend/examples/memfabric/validate_l1_backing_cpu.py \
  --core-source /absolute/path/to/worker-core \
  --ascend-source /absolute/path/to/worker-ascend
```

该入口使用真实 `L1Manager` / `StorageManager` / `TensorMemoryAllocator` 与 native TTLLock，覆盖注入 arena、写完成前不可读、读写 reservation 下关闭被拒绝、native L1 预取与读保护、释放后的 GVA 拒绝及零占用。只有 BM arena 由 CPU tensor 模拟；不初始化 SDK 或 NPU，不能替代设备验证。
