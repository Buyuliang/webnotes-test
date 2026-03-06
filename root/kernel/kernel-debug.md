# kernel-debug

```bash
setenv bootargs "earlycon console=ttyFIQ,115200 root=/dev/mmcblk0p7 rw rootwait ignore_loglevel initcall_debug"
```
启动参数添加 initcall_debug 报错会显示 call stack
