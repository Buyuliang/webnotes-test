# kernel-makefile

# Linux 内核 Makefile 详细分析文档

## 文档信息

- **内核版本**: 6.1.141
- **代码名称**: Curry Ramen
- **许可证**: GPL-2.0
- **文件路径**: `kernel/Makefile`
- **总行数**: 2161 行

---

## 目录

1. [版本信息与基本设置](#1-版本信息与基本设置)
2. [构建系统初始化](#2-构建系统初始化)
3. [输出控制机制](#3-输出控制机制)
4. [编译器与工具链配置](#4-编译器与工具链配置)
5. [架构配置](#5-架构配置)
6. [配置系统集成](#6-配置系统集成)
7. [核心构建目标](#7-核心构建目标)
8. [模块构建系统](#8-模块构建系统)
9. [清理目标](#9-清理目标)
10. [设备树支持](#10-设备树支持)
11. [Rust 支持](#11-rust-支持)
12. [打包支持](#12-打包支持)
13. [帮助系统](#13-帮助系统)
14. [关键变量与标志](#14-关键变量与标志)

---

## 1. 版本信息与基本设置

### 1.1 版本定义

```makefile
VERSION = 6
PATCHLEVEL = 1
SUBLEVEL = 141
EXTRAVERSION =
NAME = Curry Ramen
```

- **VERSION**: 主版本号
- **PATCHLEVEL**: 次版本号
- **SUBLEVEL**: 补丁级别
- **EXTRAVERSION**: 额外版本信息（通常为空）
- **NAME**: 内核代码名称

### 1.2 版本计算

```makefile
KERNELVERSION = $(VERSION)$(if $(PATCHLEVEL),.$(PATCHLEVEL)$(if $(SUBLEVEL),.$(SUBLEVEL)))$(EXTRAVERSION)
```

完整版本号格式：`6.1.141`

### 1.3 内部目标保护

```makefile
$(if $(filter __%, $(MAKECMDGOALS)), \
	$(error targets prefixed with '__' are only for internal use))
```

防止用户直接调用以 `__` 开头的内部目标。

---

## 2. 构建系统初始化

### 2.1 递归构建机制

Linux 内核使用递归构建系统，子 Makefile 只修改自己目录下的文件。

**关键特性**：
- 子 Makefile 独立工作
- 全局影响的操作在 `prepare` 阶段完成
- 支持并行构建

### 2.2 环境设置

#### 禁用内置规则

```makefile
MAKEFLAGS += -rR
```

- `-r`: 禁用内置规则
- `-R`: 禁用内置变量
- 提高性能，避免难以调试的行为

#### 区域设置

```makefile
unexport LC_ALL
LC_COLLATE=C
LC_NUMERIC=C
export LC_COLLATE LC_NUMERIC
```

确保构建过程不受区域设置影响，保证一致性。

#### Shell 环境清理

```makefile
unexport GREP_OPTIONS
```

避免 shell 环境变量干扰构建。

### 2.3 输出目录处理

#### 支持方式

1. **O= 参数**（优先级更高）
   ```bash
   make O=/path/to/output
   ```

2. **KBUILD_OUTPUT 环境变量**
   ```bash
   export KBUILD_OUTPUT=/path/to/output
   make
   ```

#### 实现机制

```makefile
ifeq ("$(origin O)", "command line")
  KBUILD_OUTPUT := $(O)
endif

ifneq ($(KBUILD_OUTPUT),)
abs_objtree := $(shell mkdir -p $(KBUILD_OUTPUT) && cd $(KBUILD_OUTPUT) && pwd)
abs_objtree := $(realpath $(abs_objtree))
else
abs_objtree := $(CURDIR)
endif
```

**特性**：
- 自动创建输出目录
- 解析符号链接
- 支持源码树外构建

#### 子 Makefile 调用

```makefile
ifeq ($(need-sub-make),1)
__sub-make:
	$(Q)$(MAKE) -C $(abs_objtree) -f $(abs_srctree)/Makefile $(MAKECMDGOALS)
endif
```

当输出目录与源码目录不同时，调用子 Makefile。

---

## 3. 输出控制机制

### 3.1 详细级别控制

#### 变量定义

```makefile
ifeq ("$(origin V)", "command line")
  KBUILD_VERBOSE = $(V)
endif
ifndef KBUILD_VERBOSE
  KBUILD_VERBOSE = 0
endif
```

#### 输出模式

| V 值 | 模式 | 说明 |
|------|------|------|
| 0 | quiet_ | 静默模式（默认），只显示简短信息 |
| 1 | 空 | 详细模式，显示完整命令 |
| 2 | - | 显示重建原因 |

#### 实现

```makefile
ifeq ($(KBUILD_VERBOSE),1)
  quiet =
  Q =
else
  quiet=quiet_
  Q = @
endif
```

- `Q = @`: 隐藏命令（静默模式）
- `Q = `: 显示命令（详细模式）

#### 静默模式支持

```makefile
ifeq ($(filter 3.%,$(MAKE_VERSION)),)
silence:=$(findstring s,$(firstword -$(MAKEFLAGS)))
else
silence:=$(findstring s,$(filter-out --%,$(MAKEFLAGS)))
endif

ifeq ($(silence),s)
quiet=silent_
KBUILD_VERBOSE = 0
endif
```

支持 `make -s` 完全静默模式。

### 3.2 源代码检查

#### Sparse 检查器

```makefile
ifeq ("$(origin C)", "command line")
  KBUILD_CHECKSRC = $(C)
endif
ifndef KBUILD_CHECKSRC
  KBUILD_CHECKSRC = 0
endif
```

**使用方式**：
- `make C=1`: 只检查重新编译的文件
- `make C=2`: 检查所有源文件

#### Rust Clippy 检查

```makefile
ifeq ("$(origin CLIPPY)", "command line")
  KBUILD_CLIPPY := $(CLIPPY)
endif
```

**使用方式**：
- `make CLIPPY=1`: 启用 Clippy 检查

### 3.3 外部模块构建

```makefile
ifeq ("$(origin M)", "command line")
  KBUILD_EXTMOD := $(M)
endif
```

**使用方式**：
```bash
make M=/path/to/module
```

**限制**：
- 不支持同时构建多个外部模块
- 路径不能包含 `%` 或 `:`

---

## 4. 编译器与工具链配置

### 4.1 架构设置

#### 架构变量

```makefile
ARCH		?= $(SUBARCH)
UTS_MACHINE 	:= $(ARCH)
SRCARCH 	:= $(ARCH)
```

#### 特殊架构映射

```makefile
# x86 架构
ifeq ($(ARCH),i386)
        SRCARCH := x86
endif
ifeq ($(ARCH),x86_64)
        SRCARCH := x86
endif

# SPARC 架构
ifeq ($(ARCH),sparc32)
       SRCARCH := sparc
endif
ifeq ($(ARCH),sparc64)
       SRCARCH := sparc
endif

# PARISC 架构
ifeq ($(ARCH),parisc64)
       SRCARCH := parisc
endif
```

#### 交叉编译检测

```makefile
export cross_compiling :=
ifneq ($(SRCARCH),$(SUBARCH))
cross_compiling := 1
endif
```

### 4.2 编译器选择

#### LLVM/Clang 支持

```makefile
ifneq ($(LLVM),)
ifneq ($(filter %/,$(LLVM)),)
LLVM_PREFIX := $(LLVM)
else ifneq ($(filter -%,$(LLVM)),)
LLVM_SUFFIX := $(LLVM)
endif

HOSTCC	= $(LLVM_PREFIX)clang$(LLVM_SUFFIX)
HOSTCXX	= $(LLVM_PREFIX)clang++$(LLVM_SUFFIX)
else
HOSTCC	= gcc
HOSTCXX	= g++
endif
```

#### 目标编译器

```makefile
ifneq ($(LLVM),)
CC		= $(LLVM_PREFIX)clang$(LLVM_SUFFIX)
LD		= $(LLVM_PREFIX)ld.lld$(LLVM_SUFFIX)
AR		= $(LLVM_PREFIX)llvm-ar$(LLVM_SUFFIX)
NM		= $(LLVM_PREFIX)llvm-nm$(LLVM_SUFFIX)
OBJCOPY		= $(LLVM_PREFIX)llvm-objcopy$(LLVM_SUFFIX)
OBJDUMP		= $(LLVM_PREFIX)llvm-objdump$(LLVM_SUFFIX)
READELF		= $(LLVM_PREFIX)llvm-readelf$(LLVM_SUFFIX)
STRIP		= $(LLVM_PREFIX)llvm-strip$(LLVM_SUFFIX)
else
CC		= $(CROSS_COMPILE)gcc
LD		= $(CROSS_COMPILE)ld
AR		= $(CROSS_COMPILE)ar
NM		= $(CROSS_COMPILE)nm
OBJCOPY		= $(CROSS_COMPILE)objcopy
OBJDUMP		= $(CROSS_COMPILE)objdump
READELF		= $(CROSS_COMPILE)readelf
STRIP		= $(CROSS_COMPILE)strip
endif
```

### 4.3 编译标志

#### C 编译器标志

```makefile
KBUILD_CFLAGS   := -Wall -Wundef -Werror=strict-prototypes -Wno-trigraphs \
		   -fno-strict-aliasing -fno-common -fshort-wchar -fno-PIE \
		   -Werror=implicit-function-declaration -Werror=implicit-int \
		   -Werror=return-type -Wno-format-security \
		   -std=gnu11
```

**关键标志**：
- `-Wall`: 启用所有警告
- `-Werror=strict-prototypes`: 严格原型检查
- `-std=gnu11`: C11 标准（GNU 扩展）
- `-fno-PIE`: 禁用位置无关可执行文件

#### Rust 编译器标志

```makefile
KBUILD_RUSTFLAGS := $(rust_common_flags) \
		    --target=$(objtree)/rust/target.json \
		    -Cpanic=abort -Cembed-bitcode=n -Clto=n \
		    -Cforce-unwind-tables=n -Ccodegen-units=1 \
		    -Csymbol-mangling-version=v0 \
		    -Crelocation-model=static \
		    -Zfunction-sections=n \
		    -Dclippy::float_arithmetic
```

#### 汇编器标志

```makefile
KBUILD_AFLAGS   := -D__ASSEMBLY__ -fno-PIE
```

### 4.4 优化选项

#### 性能优化

```makefile
ifdef CONFIG_CC_OPTIMIZE_FOR_PERFORMANCE
KBUILD_CFLAGS += -O2
KBUILD_RUSTFLAGS += -Copt-level=2
```

#### 大小优化

```makefile
else ifdef CONFIG_CC_OPTIMIZE_FOR_SIZE
KBUILD_CFLAGS += -Os
KBUILD_RUSTFLAGS += -Copt-level=s
endif
```

### 4.5 警告控制

#### 警告级别

```makefile
KBUILD_CFLAGS-$(CONFIG_WERROR) += -Werror
KBUILD_RUSTFLAGS-$(CONFIG_WERROR) += -Dwarnings
```

#### 禁用特定警告

```makefile
KBUILD_CFLAGS += $(call cc-disable-warning, unused-but-set-variable)
KBUILD_CFLAGS += $(call cc-disable-warning, unused-const-variable)
KBUILD_CFLAGS += $(call cc-disable-warning, dangling-pointer)
```

---

## 5. 架构配置

### 5.1 架构 Makefile 包含

```makefile
include $(srctree)/arch/$(SRCARCH)/Makefile
```

架构特定的配置在 `arch/$(SRCARCH)/Makefile` 中定义。

### 5.2 交叉编译

#### CROSS_COMPILE 变量

```makefile
# CROSS_COMPILE specify the prefix used for all executables used
# during compilation. Only gcc and related bin-utils executables
# are prefixed with $(CROSS_COMPILE).
```

**使用示例**：
```bash
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
```

---

## 6. 配置系统集成

### 6.1 配置目标

#### 配置目标列表

```makefile
no-dot-config-targets := $(clean-targets) \
			 cscope gtags TAGS tags help% %docs check% coccicheck \
			 $(version_h) headers headers_% archheaders archscripts \
			 %asm-generic kernelversion %src-pkg dt_binding_check \
			 outputmakefile rustavailable rustfmt rustfmtcheck
```

这些目标不需要 `.config` 文件。

#### 配置目标处理

```makefile
config: outputmakefile scripts_basic FORCE
	$(Q)$(MAKE) $(build)=scripts/kconfig $@

%config: outputmakefile scripts_basic FORCE
	$(Q)$(MAKE) $(build)=scripts/kconfig $@
```

**支持的配置目标**：
- `make config`: 文本配置
- `make menuconfig`: 菜单配置
- `make xconfig`: X11 图形配置
- `make oldconfig`: 基于旧配置更新
- `make defconfig`: 默认配置

### 6.2 配置同步

#### 自动配置同步

```makefile
%/config/auto.conf %/config/auto.conf.cmd %/generated/autoconf.h %/generated/rustc_cfg: $(KCONFIG_CONFIG)
	$(Q)$(kecho) "  SYNC    $@"
	$(Q)$(MAKE) -f $(srctree)/Makefile syncconfig
```

将 `.config` 转换为构建系统可用的头文件。

#### 配置验证

```makefile
include/config/auto.conf:
	@test -e include/generated/autoconf.h -a -e $@ || (		\
	echo >&2;							\
	echo >&2 "  ERROR: Kernel configuration is invalid.";		\
	echo >&2 "         include/generated/autoconf.h or $@ are missing.";\
	echo >&2 "         Run 'make oldconfig && make prepare' on kernel src to fix it.";	\
	echo >&2 ;							\
	/bin/false)
```

---

## 7. 核心构建目标

### 7.1 默认目标

```makefile
PHONY += all
ifeq ($(KBUILD_EXTMOD),)
__all: all
else
__all: modules
endif

all: vmlinux
```

- 内核构建：`all` → `vmlinux`
- 外部模块：`__all` → `modules`

### 7.2 模块与内置对象

#### 构建模式

```makefile
KBUILD_MODULES :=
KBUILD_BUILTIN := 1

# 如果只有 "make modules"，不编译内置对象
ifeq ($(MAKECMDGOALS),modules)
  KBUILD_BUILTIN :=
endif

# 如果包含 "make <whatever> modules"，编译模块
ifneq ($(filter all modules nsdeps %compile_commands.json clang-%,$(MAKECMDGOALS)),)
  KBUILD_MODULES := 1
endif

# 默认情况下也构建模块
ifeq ($(MAKECMDGOALS),)
  KBUILD_MODULES := 1
endif
```

### 7.3 vmlinux 构建流程

#### 构建步骤

1. **vmlinux.a** - 归档所有对象文件
   ```makefile
   vmlinux.a: $(KBUILD_VMLINUX_OBJS) scripts/head-object-list.txt autoksyms_recursive FORCE
   	$(call if_changed,ar_vmlinux.a)
   ```

2. **vmlinux_o** - 链接对象文件
   ```makefile
   vmlinux_o: vmlinux.a $(KBUILD_VMLINUX_LIBS)
   	$(Q)$(MAKE) -f $(srctree)/scripts/Makefile.vmlinux_o
   ```

3. **vmlinux** - 最终链接
   ```makefile
   vmlinux: vmlinux.o $(KBUILD_LDS) modpost
   	$(Q)$(MAKE) -f $(srctree)/scripts/Makefile.vmlinux
   ```

#### 自动符号修剪

```makefile
ifdef CONFIG_TRIM_UNUSED_KSYMS
autoksyms_recursive: $(build-dir) modules.order
	$(Q)$(CONFIG_SHELL) $(srctree)/scripts/adjust_autoksyms.sh \
	  "$(MAKE) -f $(srctree)/Makefile autoksyms_recursive"
endif
```

自动移除未使用的导出符号。

### 7.4 准备阶段

#### 准备层次

```makefile
archprepare: outputmakefile archheaders archscripts scripts include/config/kernel.release \
	asm-generic $(version_h) $(autoksyms_h) include/generated/utsrelease.h \
	include/generated/compile.h include/generated/autoconf.h remove-stale-files

prepare0: archprepare
	$(Q)$(MAKE) $(build)=scripts/mod
	$(Q)$(MAKE) $(build)=. prepare

prepare: prepare0
ifdef CONFIG_RUST
	$(Q)$(CONFIG_SHELL) $(srctree)/scripts/rust_is_available.sh
	$(Q)$(MAKE) $(build)=rust
endif
```

**准备阶段顺序**：
1. `archprepare`: 架构相关准备
2. `prepare0`: 模块和通用准备
3. `prepare`: Rust 支持（如果启用）

---

## 8. 模块构建系统

### 8.1 模块准备

```makefile
modules: modules_prepare

modules_prepare: prepare
	$(Q)$(MAKE) $(build)=scripts scripts/module.lds
```

### 8.2 模块安装

```makefile
modules_install: $(modinst_pre)
__modinst_pre:
	@rm -rf $(MODLIB)/kernel
	@rm -f $(MODLIB)/source
	@mkdir -p $(MODLIB)/kernel
	@ln -s $(abspath $(srctree)) $(MODLIB)/source
	@if [ ! $(objtree) -ef  $(MODLIB)/build ]; then \
		rm -f $(MODLIB)/build ; \
		ln -s $(CURDIR) $(MODLIB)/build ; \
	fi
	@sed 's:^:kernel/:' modules.order > $(MODLIB)/modules.order
	@cp -f modules.builtin $(MODLIB)/
	@cp -f $(objtree)/modules.builtin.modinfo $(MODLIB)/
```

**安装位置**：
- 默认：`/lib/modules/$(KERNELRELEASE)/`
- 可通过 `INSTALL_MOD_PATH` 指定前缀

### 8.3 模块签名

```makefile
ifeq ($(CONFIG_MODULE_SIG),y)
PHONY += modules_sign
modules_sign: modules_install
	@:
endif
```

---

## 9. 清理目标

### 9.1 清理级别

#### make clean

```makefile
CLEAN_FILES += include/ksym vmlinux.symvers modules-only.symvers \
	       modules.builtin modules.builtin.modinfo modules.nsdeps \
	       compile_commands.json rust/test rust/doc \
	       .vmlinux.objs .vmlinux.export.c

clean: archclean vmlinuxclean resolve_btfids_clean
```

**功能**：
- 删除大部分生成文件
- 保留配置和构建外部模块所需文件

#### make mrproper

```makefile
MRPROPER_FILES += include/config include/generated          \
		  arch/$(SRCARCH)/include/generated .objdiff \
		  debian snap tar-install \
		  .config .config.old .version \
		  Module.symvers \
		  certs/signing_key.pem \
		  certs/x509.genkey \
		  vmlinux-gdb.py \
		  *.spec \
		  rust/target.json rust/libmacros.so

mrproper: clean $(mrproper-dirs)
	$(call cmd,rmfiles)
```

**功能**：
- 删除所有生成文件
- 删除配置文件（`.config`）

#### make distclean

```makefile
distclean: mrproper
	@find . $(RCS_FIND_IGNORE) \
		\( -name '*.orig' -o -name '*.rej' -o -name '*~' \
		-o -name '*.bak' -o -name '#*#' -o -name '*%' \
		-o -name 'core' -o -name tags -o -name TAGS -o -name 'cscope*' \
		-o -name GPATH -o -name GRTAGS -o -name GSYMS -o -name GTAGS \) \
		-type f -print | xargs rm -f
```

**功能**：
- `mrproper` + 删除编辑器备份文件

---

## 10. 设备树支持

### 10.1 设备树构建

```makefile
%.dtb: dtbs_prepare
	$(Q)$(MAKE) $(build)=$(dtstree) $(dtstree)/$@

%.dtbo: dtbs_prepare
	$(Q)$(MAKE) $(build)=$(dtstree) $(dtstree)/$@

dtbs: dtbs_prepare
	$(Q)$(MAKE) $(build)=$(dtstree)
```

### 10.2 设备树安装

```makefile
dtbs_install:
	$(Q)$(MAKE) $(dtbinst)=$(dtstree) dst=$(INSTALL_DTBS_PATH)
```

**安装位置**：
- 默认：`$(INSTALL_PATH)/dtbs/$(KERNELRELEASE)`
- 可通过 `INSTALL_DTBS_PATH` 指定

### 10.3 设备树验证

```makefile
dt_binding_check: scripts_dtc
	$(Q)$(MAKE) $(build)=Documentation/devicetree/bindings

dtbs_check: dtbs
```

---

## 11. Rust 支持

### 11.1 Rust 工具检查

```makefile
PHONY += rustavailable
rustavailable:
	$(Q)$(CONFIG_SHELL) $(srctree)/scripts/rust_is_available.sh && echo "Rust is available!"
```

### 11.2 Rust 格式化

```makefile
rustfmt:
	$(Q)find $(abs_srctree) -type f -name '*.rs' \
		-o -path $(abs_srctree)/rust/alloc -prune \
		-o -path $(abs_objtree)/rust/test -prune \
		| grep -Fv $(abs_srctree)/rust/alloc \
		| grep -Fv $(abs_objtree)/rust/test \
		| grep -Fv generated \
		| xargs $(RUSTFMT) $(rustfmt_flags)

rustfmtcheck: rustfmt_flags = --check
rustfmtcheck: rustfmt
```

### 11.3 Rust 文档

```makefile
PHONY += rustdoc
rustdoc: prepare
	$(Q)$(MAKE) $(build)=rust $@
```

### 11.4 Rust 测试

```makefile
PHONY += rusttest
rusttest: prepare
	$(Q)$(MAKE) $(build)=rust $@
```

---

## 12. 打包支持

### 12.1 打包目标

```makefile
%src-pkg: FORCE
	$(Q)$(MAKE) -f $(srctree)/scripts/Makefile.package $@

%pkg: include/config/kernel.release FORCE
	$(Q)$(MAKE) -f $(srctree)/scripts/Makefile.package $@
```

### 12.2 支持的打包格式

通过 `scripts/Makefile.package` 支持：

- **RPM**: `make rpm-pkg`, `make binrpm-pkg`
- **Deb**: `make deb-pkg`, `make bindeb-pkg`
- **Tarball**: `make tar-pkg`, `make targz-pkg`, `make tarbz2-pkg`, `make tarxz-pkg`, `make tarzst-pkg`
- **Snap**: `make snap-pkg`
- **目录**: `make dir-pkg`

---

## 13. 帮助系统

### 13.1 帮助目标

```makefile
PHONY += help
help:
	@echo  'Cleaning targets:'
	@echo  'Configuration targets:'
	@echo  'Other generic targets:'
	@echo  'Static analysers:'
	@echo  'Tools:'
	@echo  'Kernel selftest:'
	@echo  'Rust targets:'
	@echo  'Devicetree:'
	@echo  'Kernel packaging:'
	@echo  'Documentation targets:'
	@echo  'Architecture specific targets:'
```

### 13.2 使用说明

帮助系统显示：
- 清理目标
- 配置目标
- 构建目标
- 静态分析工具
- 测试目标
- 打包选项
- 架构特定目标

---

## 14. 关键变量与标志

### 14.1 构建变量

| 变量 | 说明 |
|------|------|
| `KBUILD_VERBOSE` | 详细级别 (0/1/2) |
| `KBUILD_CHECKSRC` | 源代码检查级别 (0/1/2) |
| `KBUILD_EXTMOD` | 外部模块目录 |
| `KBUILD_OUTPUT` | 输出目录 |
| `KBUILD_MODULES` | 是否构建模块 |
| `KBUILD_BUILTIN` | 是否构建内置对象 |

### 14.2 编译器变量

| 变量 | 说明 |
|------|------|
| `CC` | C 编译器 |
| `LD` | 链接器 |
| `AR` | 归档工具 |
| `NM` | 符号列表工具 |
| `OBJCOPY` | 对象复制工具 |
| `OBJDUMP` | 对象转储工具 |
| `STRIP` | 符号剥离工具 |

### 14.3 编译标志变量

| 变量 | 说明 |
|------|------|
| `KBUILD_CFLAGS` | C 编译器标志 |
| `KBUILD_AFLAGS` | 汇编器标志 |
| `KBUILD_RUSTFLAGS` | Rust 编译器标志 |
| `KBUILD_LDFLAGS` | 链接器标志 |
| `KBUILD_CPPFLAGS` | 预处理器标志 |

### 14.4 路径变量

| 变量 | 说明 |
|------|------|
| `srctree` | 源码树根目录 |
| `objtree` | 对象树根目录 |
| `abs_srctree` | 绝对源码路径 |
| `abs_objtree` | 绝对对象路径 |
| `VPATH` | 虚拟路径 |

### 14.5 安装路径变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `INSTALL_PATH` | 内核安装路径 | `/boot` |
| `INSTALL_MOD_PATH` | 模块安装前缀 | 空 |
| `INSTALL_HDR_PATH` | 头文件安装路径 | `$(objtree)/usr` |
| `INSTALL_DTBS_PATH` | 设备树安装路径 | `$(INSTALL_PATH)/dtbs/$(KERNELRELEASE)` |

---

## 15. 常用构建命令

### 15.1 基本构建

```bash
# 默认构建（vmlinux + 模块）
make

# 只构建内核
make vmlinux

# 只构建模块
make modules

# 构建并安装模块
make modules_install
```

### 15.2 配置相关

```bash
# 交互式配置
make menuconfig

# 基于默认配置
make defconfig

# 更新配置
make oldconfig

# 查看配置帮助
make helpconfig
```

### 15.3 清理

```bash
# 清理生成文件
make clean

# 完全清理（包括配置）
make mrproper

# 深度清理（包括备份文件）
make distclean
```

### 15.4 交叉编译

```bash
# ARM64 交叉编译
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-

# 指定输出目录
make O=/path/to/output ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
```

### 15.5 调试构建

```bash
# 详细输出
make V=1

# 显示重建原因
make V=2

# 源代码检查
make C=1

# 启用额外警告
make W=1
```

### 15.6 打包

```bash
# Debian 包
make bindeb-pkg

# RPM 包
make binrpm-pkg

# Tarball
make targz-pkg
```

---

## 16. 构建流程总结

### 16.1 完整构建流程

```
1. 初始化阶段
   ├── 环境设置
   ├── 输出目录处理
   └── 子 Makefile 调用（如需要）

2. 配置阶段
   ├── 读取 .config
   ├── 同步配置到头文件
   └── 生成 autoconf.h

3. 准备阶段
   ├── archprepare
   ├── prepare0
   └── prepare

4. 构建阶段
   ├── 构建核心对象 (core-y)
   ├── 构建驱动 (drivers-y)
   ├── 构建库 (libs-y)
   ├── 归档为 vmlinux.a
   ├── 链接为 vmlinux.o
   └── 最终链接为 vmlinux

5. 模块阶段（如果启用）
   ├── 构建模块对象
   ├── 链接模块
   └── 生成 .ko 文件

6. 安装阶段（可选）
   ├── 安装内核镜像
   ├── 安装模块
   └── 安装设备树
```

### 16.2 关键依赖关系

```
vmlinux
  ├── vmlinux.o
  │   └── vmlinux_o
  │       ├── vmlinux.a
  │       │   └── $(KBUILD_VMLINUX_OBJS)
  │       └── $(KBUILD_VMLINUX_LIBS)
  ├── $(KBUILD_LDS)
  └── modpost

modules
  └── modules_prepare
      └── prepare
          └── prepare0
              └── archprepare
```

---

## 17. 特殊功能

### 17.1 单文件构建

```makefile
single-targets := %.a %.i %.rsi %.ko %.lds %.ll %.lst %.mod %.o %.s %.symtypes %/
```

**使用示例**：
```bash
# 构建单个文件
make drivers/gpu/arm/bifrost/mali_kbase.o

# 构建单个模块
make drivers/gpu/arm/bifrost/mali_kbase.ko
```

### 17.2 混合目标

```makefile
ifdef mixed-build
__build_one_by_one:
	$(Q)set -e; \
	for i in $(MAKECMDGOALS); do \
		$(MAKE) -f $(srctree)/Makefile $$i; \
	done
endif
```

支持同时指定配置目标和构建目标，例如：
```bash
make oldconfig all
```

### 17.3 版本信息生成

```makefile
filechk_kernel.release = \
	echo "$(KERNELVERSION)$$($(CONFIG_SHELL) $(srctree)/scripts/setlocalversion $(srctree))"
```

自动生成包含 Git 信息的版本字符串。

---

## 18. 性能优化特性

### 18.1 增量构建

- 基于时间戳的依赖检查
- 只重新编译修改的文件
- 并行构建支持（`-j` 选项）

### 18.2 缓存优化

- 编译命令缓存
- 依赖文件缓存（`.d` 文件）
- 符号表缓存

### 18.3 并行构建

```bash
# 使用所有 CPU 核心
make -j$(nproc)

# 指定并行数
make -j8
```

---

## 19. 调试与诊断

### 19.1 调试目标

```makefile
# 生成调试信息
CONFIG_DEBUG_INFO=y

# 生成 GDB 脚本
scripts_gdb: prepare0
	$(Q)$(MAKE) $(build)=scripts/gdb
	$(Q)ln -fsn $(abspath $(srctree)/scripts/gdb/vmlinux-gdb.py)
```

### 19.2 诊断工具

- `make checkstack`: 检查栈使用
- `make versioncheck`: 版本检查
- `make includecheck`: 头文件包含检查
- `make coccicheck`: Coccinelle 检查

---

## 20. 总结

### 20.1 核心特性

1. **递归构建系统**: 支持大规模项目的模块化构建
2. **灵活的配置系统**: 与 Kconfig 深度集成
3. **多架构支持**: 支持多种 CPU 架构和交叉编译
4. **模块化设计**: 清晰的内置对象和模块分离
5. **丰富的工具链支持**: GCC、Clang、Rust
6. **完善的清理机制**: 多级清理目标
7. **打包支持**: 支持多种发行版打包格式

### 20.2 设计原则

1. **向后兼容**: 保持与旧版本的兼容性
2. **可扩展性**: 易于添加新功能和目标
3. **性能优先**: 优化构建速度和增量构建
4. **用户友好**: 提供详细的帮助和错误信息

### 20.3 最佳实践

1. 使用 `O=` 进行源码树外构建
2. 使用 `V=1` 调试构建问题
3. 使用 `-j` 进行并行构建
4. 定期运行 `make clean` 保持构建目录整洁
5. 使用 `make help` 查看可用目标

---

## 附录

### A. 相关文件

- `scripts/Kbuild.include`: Kbuild 通用函数
- `scripts/Makefile.build`: 构建规则
- `scripts/Makefile.clean`: 清理规则
- `scripts/Makefile.modpost`: 模块后处理
- `scripts/Makefile.package`: 打包规则

### B. 相关文档

- `Documentation/kbuild/`: Kbuild 系统文档
- `README`: 内核构建说明
- `Documentation/dev-tools/`: 开发工具文档

### C. 版本历史

- 本文档基于 Linux 6.1.141 内核
- Makefile 结构在主要版本间保持稳定
- 新功能通过向后兼容的方式添加

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: 内核开发团队
