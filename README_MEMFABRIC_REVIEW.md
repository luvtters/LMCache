# worker 直传的 LMCache core 配套修改

**独立分支设备复验：NOT_VERIFIED。** 以下设备记录只属于原运行快照；源码提取和本次 CPU 通过不能替代独立分支上机复验。

配套 Ascend：luvtters/LMCache-Ascend 的 `review/memfabric-worker-direct`。L2 兼容路线不需要本分支的扩展。

基线是官方 `6913259264dbebb2fa5ba1665606c40ffc5c6661`；全部十个实现、测试、说明文件原样提取自已使用的 `5b565d211416cd48bebf20bae12a4abc15a72ba9`，没有加入后续 L1 allocator/批量注册候选。

## 差异

- Connector 新增可覆写 scheduler adapter 工厂，默认仍用原 adapter。
- scheduler 把 lookup state 构造提取为 hook，让直传的查询和锁保护使用同一个池目录。
- transfer_channel 改为实际创建时才加载可选 native 后端，避免仅导入 RPC 类型就加载无关 SDK。
- 回调关闭增加存活检查和幂等 quiesce，未知完成不提前释放上下文。

本分支没有 MemFabric SDK、池索引或搬运实现；这些位于配套 Ascend 分支。它不是整体重写 LMCache L1，也不表示 worker 已完成 L1 兼容。

## 验证边界

该运行基线曾用于 Qwen3-8B worker 直传 TP1/TP2/TP4 真实功能检查；原运行仍有退出异常，不能称生命周期已全部通过。本次仅重新运行定向 CPU 检查，不重新上机。

```sh
python tests/v1/test_mp_adapter_extension_hooks.py
python tests/v1/distributed/transfer_channel/test_lazy_import.py
python tests/v1/multiprocess/check_memfabric_callback_shutdown_cpu.py
```
