# note001

```bash
1、查看是否包含 npu 驱动：
 zcat /proc/config.gz | grep RKNPU

2、可以使用 https://github.com/airockchip/rknn_model_zoo 的 demo 测试

3、查看 npu 负载
sudo cat /sys/kernel/debug/rknpu/load
```

