# TransformerEngine-Plugin-FL

TransformerEngine 的算子分发插件:把 `transformer_engine_torch`(原本是 NVIDIA 的 pybind 扩展)
替换成一个纯 Python 的分发层,让 FlagOS、各厂商扩展、PyTorch 参考实现可以按策略挑选,
而上层的 `transformer_engine.pytorch` 代码一行都不用改。

## 1. 整体设计

### 1.1 两个仓库的分工

| 仓库 | 角色 | 内容 |
| --- | --- | --- |
| `TransformerEngine`(分支 `cuda_patch_main_latest`) | **接口** | fork 自上游 main,只加了插件挂载点 |
| `TransformerEngine-Plugin-FL`(本仓库) | **实现** | 独立的 `transformer_engine_plugin_fl` 包 |

`TransformerEngine-FL` 是**旧设计**(把插件系统内嵌在 `transformer_engine/plugin/core/` 里,
且 fork 自一个较老的 release 分支),已经被上面这两个仓库取代。本仓库的算子 API 以上游
**main** 为准,不以 `TransformerEngine-FL` 为准 —— 后者的签名已经和 main 分叉了。

### 1.2 挂载流程

接口仓库里只加了十几行(`transformer_engine/common/__init__.py`)。设置
`NVTE_PLUGIN=transformer_engine_plugin_fl` 后:

```
1. sys.modules["transformer_engine_torch_nv"] = 真正的 pybind .so   ← 先做别名
2. import transformer_engine_plugin_fl → load_plugins()
3. sys.modules["transformer_engine_torch"] = stub(纯 Python)
4. transformer_engine_plugin_fl.patches.apply_patches()            ← 可选,最后做
```

第 1 步必须在 import 插件之前,因为 CUDA 后端会在模块顶层 import
`transformer_engine_torch_nv`。第 4 步放最后,因为设备 patch 会原地改写进程全局状态、
无法回滚,所以 `load_plugins()` 不能依赖 patch 之后的状态。
插件初始化失败时接口仓库会回滚到挂载前的状态。

### 1.3 运行时分发

```
transformer_engine.pytorch 调用 tex.rmsnorm_fwd(...)
        ↓
stub.py            （sys.modules["transformer_engine_torch"]）
        ↓
OpManager.call()   按 policy 选实现,失败则 fallback 到下一候选
        ↓
OpRegistry         每个 op 挂着多个 OpImpl,按 kind/priority 排序
        ↓
FlagOSBackend / CUDABackend / MetaxBackend / ReferenceBackend ...
        ↓
flag_gems / transformer_engine_torch_nv / transformer_engine_torch_metax / torch
```

核心模块:`registry.py`(注册表)、`policy.py`(选择策略)、`manager.py`(分发+fallback)、
`stub.py`(伪装成 pybind 模块)、`ops.py`(算子 API 基类)。

## 2. 各后端的 ops 覆盖范围

`ops.py` 的 `TEFLBackendBase` 是**唯一的接口基准**,共 193 个方法(192 个可分发 op +
`is_available`),严格对齐 TransformerEngine main 的 pybind surface。

| 后端 | 覆盖 | 实现方式 | 升级原则 |
| --- | --- | --- | --- |
| `ops.py` (`TEFLBackendBase`) | **193** | `raise NotImplementedError` | **全量**,对齐 main |
| `vendor/cuda` | **193** | 直接 `tex.xxx(...)` | **全量**,对齐 main |
| `vendor/enflame` | **193** | 直接 `tex.xxx(...)` | **全量**,但保留厂商微调 |
| `vendor/hygon` | **193** | 直接 `tex.xxx(...)` | 同上 |
| `vendor/iluvatar` | **193** | 直接 `tex.xxx(...)` | 同上 |
| `vendor/metax` | **193** | 直接 `tex.xxx(...)` | 同上 |
| `vendor/musa` | **193** | 直接 `tex.xxx(...)` | 同上 |
| `flagos` | 27 | `flag_gems` / Triton | **子集**,只迁已有 api |
| `reference` | 66 | 纯 PyTorch | **子集**,只迁已有 api |
| `vendor/kunlunxin` | 21 | 厂商扩展 | **子集**,只迁已有 api |
| `vendor/npu` | 15 | `torch_npu` | **子集**,只迁已有 api |

op 名单集中在 `backends/vendor/op_names.py`:

- `TEX_OPS`(172)— 普通 pybind 算子
- `OPTIONAL_OPS`(11)— `ep_*`,上游在 `NVTE_WITH_NCCL_EP` 下才编译。**声明但按需注册**:
  加载的 `.so` 没有这个符号时自动跳过,所以没开 EP 编译的 `.so` 也能正常加载
- `PLUGIN_OPS`(8)— 插件内部钩子(`create_comm_overlap*`、`get_flash_attention_class` 等),
  不是 pybind 算子,但走同一套分发
- `SUBMODULE_OPS`(1)— `grouped_mlp_experimental` 子模块里的算子,由 `stub.py` 重新挂回子模块

### 2.1 为什么分"全量"和"子集"两类

**全量后端**(cuda + 5 个透传厂商)背后是和上游同源编译的扩展,所以每个 op 都应该能转发。
缺一个 op 就意味着这条路径只能 fallback 到 reference,性能悬崖式下降。

**子集后端**(flagos / reference / kunlunxin / npu)是自己实现算子的,没实现的 op
由 manager 自动 fallback 到别的后端。所以它们**只需要迁移 `TransformerEngine-FL` 里已有的
api**,不用补全 —— 但接口变化必须处理(见 §3.2),后续按需由用户新增。

## 3. 升级流程

### 3.1 对齐 ops.py(全量,跟随 main)

上游 main 一动,先重新提取它的 pybind surface:

1. 扫 `transformer_engine/pytorch/csrc/**/*.cpp` 和 `common/util/pybind_helper.h`
2. **先把 `\<换行>` 续行去掉** —— `pybind_helper.h` 在多行宏里定义算子,
   不做这步会静默漏算子(曾经漏过 `get_stream_priority_range`)
3. `m.def(` 经常跨行,单行正则会少算;匹配到名字后要**配对括号**走完参数列表
4. 参数列表:有 `py::arg(...)` 注解的直接读;没有的(约一半)去 `pytorch/csrc/extensions.h`
   看 C++ 声明。`py::kw_only()` 对应 Python 的 `*` 分隔符
5. 改 `ops.py`,再跑 §3.5 的校验

签名**严格对齐 main**,不保留旧 release 分支的参数名。

### 3.2 子集后端:处理接口变化(flagos / reference / kunlunxin / npu)

这四个后端只迁移 `TransformerEngine-FL` 中已有的 api,但要处理两类变化:

**(a) 签名变化** —— 比如 `get_fused_attn_backend` 从 19 个平铺参数收成了一个
`fused_attn_params` 对象,返回值也变成 `(backend, message)` 元组。

**(b) 签名变化带来的实现变化** —— 这是容易漏的。新增参数不是加上就完事,
函数体的语义也得跟着改。例如 `clamped_swiglu` 新增 `glu_linear_offset` 后,
reference 实现里原本硬编码的 `+ 1` 必须换成 `+ glu_linear_offset`:

```python
# 之前:CUDA 把 b 夹到 [-limit, limit] 再加 1
b_clamped = torch.clamp(b, -limit, limit) + 1
# 之后:偏移量由参数给出
b_clamped = torch.clamp(b, -limit, limit) + glu_linear_offset
```

**(c) 算子删除** —— main 里已经不存在的 op 要删掉,否则会注册进分发表却永远不被调用
(本次清理了 kunlunxin 的 `cast_to_fp8`)。

### 3.3 全量厂商后端:生成 + 保留微调

enflame / hygon / iluvatar / metax / musa 这五个是纯透传,约 187 个 op 的函数体和 CUDA
后端逐字相同(只是 `_get_tex()` 返回的扩展不同)。所以它们的算子区由
`tools/gen_vendor_ops.py` **从 CUDA 后端生成**,夹在标记之间:

```python
    ##### BEGIN GENERATED OPS (tools/gen_vendor_ops.py) #####
    ...
    ##### END GENERATED OPS #####
```

用法:

```bash
python tools/gen_vendor_ops.py              # 重新生成全部(跳过 cuda 和子集后端)
python tools/gen_vendor_ops.py metax        # 只生成 metax
python tools/gen_vendor_ops.py --check      # 只检查漂移,有漂移则非零退出(适合放 CI)
```

**标记区外的代码永远不会被生成覆盖**,包括:

- 库加载(`_load_xxx_libs`、`_check_xxx_available`、`_get_tex`)
- `is_available` / `check_available` / `get_attention_backend` / `get_flash_attention_class`
- 该厂商在 `register_ops.py` 之外主动 override 的 op

`TransformerEngine-FL` 里厂商做过的**轻微改动必须同步过来**,方式就是把它从生成区挪出去
(在 `tools/gen_vendor_ops.py` 的 `KEEP`/`HAND_WRITTEN` 语义下自动排除)。本次保留的真实差异:

| op | 差异 | 涉及 |
| --- | --- | --- |
| `multi_tensor_scale_tensor` | 厂商扩展只有标量版,降级调 `multi_tensor_scale(scale.item())` | hygon、metax |
| `create_comm_overlap` | 厂商不支持 cuBLASMp,`use_cublasmp=True` 时 `raise NotImplementedError` | 全部 5 家 |
| `create_comm_overlap_p2p` | 同上 | 全部 5 家 |
| `get_attention_backend` | 各家走自己的 dpa utils | 各不相同 |
| `get_flash_attention_class` | 各家自己的 FlashAttention 类 | 各不相同 |

判断方法:如果 5 家写法一致但和 CUDA 不同,那是**跟 main 的漂移**,应该取 CUDA 的版本;
如果 5 家之间互不相同,那是**真实的硬件差异**,必须保留。

### 3.4 带 patches 的厂商

有些厂商需要 Python 侧的设备补丁(把 `torch.cuda.*` 重定向到自家设备等)。
这类后端迁移时**除了算子,还要迁 `patches.py` 并挂进分发**:

```
backends/vendor/musa/patches.py
backends/vendor/npu/patches.py
backends/vendor/tsingmicro/patches.py     ← 只有 patches,没有后端(算子走 FlagOS)
```

`patches/__init__.py` 用名单遍历,新增厂商只要往 `PATCHED_VENDORS` 加一项:

```python
PATCHED_VENDORS = ("musa", "npu", "tsingmicro")
```

每个 `patches.py` 自己判断硬件在不在(不在就直接 return),因为一台机器上只会有一家生效。
注意 `tsingmicro` 是只有 patches 的特例:它用 FlagOS 的算子实现,所以 `vendor/tsingmicro/`
下没有后端类,但仍需要 `__init__.py` 才能被 import。

### 3.5 校验

```bash
# 1. 厂商生成区有没有漂移
python tools/gen_vendor_ops.py --check

# 2. 全部 10 个后端的签名是否与基类一致(比参数名/顺序/kind,不比类型注解)
python -m pytest transformer_engine_plugin_fl/tests/test_op_api_alignment.py -q

# 3. 整体
python -m pytest transformer_engine_plugin_fl/tests/ -q
```

`test_op_api_alignment.py` 是关键守卫。**必须比对全部 10 个后端**:只比 `ops.py` 和 CUDA
会漏掉真实问题 —— 改了基类签名,会让其他后端已经 override 的那个 op 静默失配,
而这种错误在 import 时不报,只在真正调用时才炸。

> 注意:插件加载是**静默降级**的。某个后端注册失败只会被 `except` 吞掉(因为"这台机器上没有
> 这家硬件"是正常情况)。所以升级后一定要确认 op 总数,不能只看"没报错"。
> 本次就是这样发现 reference 后端整体注册失败的:`DType` 在 Python 侧没有 `kInt16`/`kInt64`
> (C++ enum 里有,但 `pybind_helper.h` 没绑定),dtype 映射表引用它们直接抛异常,
> 结果 op 数从 70 掉到 61 而全程没有任何报错。

本地没有厂商硬件时,可以注入一个假的扩展模块来验证分发链路(注册数量、EP op 跳过、
fallback 顺序),这能覆盖接口层;数值正确性仍需在真实卡上验证。

## 4. 本次升级做了什么

### 4.1 迁移

- 从 `TransformerEngine-FL` 迁入:`npu` 后端(含 patches、flash_attention)、
  `tsingmicro/patches.py`、flagos 的 Triton RoPE(`impl/trition/`)及 7 个方法、
  reference 的 `context_parallel.py` 及 5 个 op、`fa_utils.py`
- 同步了 `TransformerEngine-FL` 2026-06 之后的修复:reference gemm 的 backward-GELU/BGRAD
  语义、adam 的 `inv_scale` + master_params、compute_scale 语义、metax 的
  `TE_METAX_TE_HOME` 新版包布局、hygon 的 `multi_tensor_scale_tensor` 双接口回退
- 5 个透传厂商后端从 138 补到全量 193 个 op

### 4.2 修掉的真实缺陷

- **`tex.CommOverlap` 参数错位**:pybind 构造函数把 `use_cublasmp`/`comm_type` 放在第 5-6 位,
  在 `num_splits` **之前**。按位置透传会把 `num_splits` 塞进 `use_cublasmp`。
  改成按关键字转发。另外 main 的 `module/base.py` 确实会传这两个参数,而插件之前完全没声明
- **reference 后端整体注册失败**:`DType.kInt16`/`kInt64` 在 Python 侧不存在(见 §3.5)
- **`get_grouped_gemm_setup_workspace_size`**:参数名应为 `num_tensors`,
  且函数体里用的还是旧名 `num_gemms`,实际是 NameError
- **`cast_to_fp8`**:main 里已删除,从 kunlunxin 移除

### 4.3 结构清理

- 删除 `backends/` 下 7 个扁平厂商目录(26 个 `.py`),它们是 `backends/vendor/*` 的旧副本,
  用的还是旧类名 `BackendBase`、import 已经失效
- `register_ops.py` 从手写 `OpImpl` 改为名单生成:5 个厂商文件 5796 行 → 120 行
  (同一厂商的所有条目 `impl_id`/`vendor`/`priority` 都一样,逐条手写不携带任何信息)
- 共享逻辑抽到 `vendor/registration.py` 和 `vendor/op_names.py`
- `patches/__init__.py` 改为名单遍历,支持 npu / tsingmicro

### 4.4 结果

- 10 个后端全部 **0 签名漂移**
- `test_op_api_alignment.py` 28/28 通过
- 用假扩展验证:181 个 op 注册、EP op 在符号缺失时正确跳过、
  flagos→vendor.cuda fallback 正常、metax/hygon 的 scale 降级与 cuBLASMp 拒绝符合预期

## 5. 已知问题

**8 个 `test_policy.py` 失败(先于本次升级存在)**:测试用 `NVTE_*` 环境变量,
代码读 `TE_FL_*`。`NVTE_*` 是目标命名(上游 TE 就是这个前缀,接口仓库也已在用 `NVTE_PLUGIN`),
但 `FlagScale` 当前在写 `TE_FL_*`:

- `FlagScale/flagscale/train/megatron/backend_config.py` 设置 `TE_FL_PREFER`、
  `TE_FL_PER_OP`、`TE_FL_ALLOW_VENDORS`、`TE_FL_DENY_VENDORS`

单独改插件会直接破坏 FlagScale,因此**留到 `FlagScale` + `Megatron-LM-FL` + 插件联合升级时
一次性改名**。在那之前这 8 个失败是预期的 —— 不要为了让它变绿而把测试改成 `TE_FL_*`,
那会把错误的命名固化下来。

待改名清单(`TE_FL_` → `NVTE_`):`STRICT`、`PREFER`、`PREFER_VENDOR`、`ALLOW_VENDORS`、
`DENY_VENDORS`、`PER_OP`、`PLUGIN_MODULES`。建议过渡期内同时接受两种前缀。
(`SKIP_CUDA`/`SKIP_HYGON`/`SKIP_METAX` 已在本次处理,见 §5.1。)

**`FlagScale` 的 policy reset 是死代码**:`backend_config.py:52` 查的是
`sys.modules.get("transformer_engine.plugin.core.policy")` —— 那是旧内嵌设计的路径,
新架构下应为 `transformer_engine_plugin_fl.policy`,TE 仓里已经没有 `transformer_engine/plugin/`。
所以设完 `TE_FL_*` 之后的 `reset_global_policy()` 永远拿不到函数,静默跳过。
目前不出问题,是因为 `configure_backend_environment` 跑在 `validate_args`/`initialize` 阶段,
早于任何 TE dispatch,policy 还没被 lazily 缓存。但一旦有代码在它之前触发首次 dispatch,
环境变量就会失效且无任何提示 —— 和上面的改名一起修。

### 5.1 skip-cuda 开关已收敛为 `NVTE_PLUGIN_SKIP_CUDA`

原先有四套重叠的机制,现在只剩一个。已删除:

| 删除项 | 原因 |
| --- | --- |
| `_build_config.py` 的三处 import | 该文件是**旧内嵌架构**(`transformer_engine/plugin/core/`)构建时生成的。本仓 `setup.py` 从未生成过它(`git log -S` 确认),所以三处 `try: from ..._build_config import ...` 必然 `ImportError`、永远走 except 分支,属于死代码。它还每次 import 打印一行 `Build config not found` 噪音 |
| `TE_FL_SKIP_CUDA` | 改名为 `NVTE_PLUGIN_SKIP_CUDA`,与接口仓 `transformer_engine/common/plugin.py` 统一 |
| `TE_FL_SKIP_HYGON` / `TE_FL_SKIP_METAX` | 不需要:`is_available()` 本来就会探测厂商扩展在不在,探测失败即跳过 |
| `backends/vendor/__init__.py` 的 eager import 门控 | 没有实际作用:`builtin_ops.py` 直接 import 各厂商的 `register_ops`,后者无条件 `from .xxx import XxxBackend`,所以在这里少做一次 eager import 挡不住任何东西。真正的闸门是 `registration.py:53` 的 `if not backend.is_available(): return 0` |
| `get_vendor_loading_errors()` | 它的数据源只有上面那个被删掉的 try/except 块,保留会变成恒返回 `[]` 的假 API;全仓无调用者 |

`_build_config` 还藏了一个**优先级矛盾**:`vendor/__init__.py` 是环境变量优先,
而 `cuda.py` 是构建配置优先(只在 `ImportError` 时才看环境变量)。旧设计的
`skip_cuda_build()` 明确是环境变量优先("allows runtime override"),所以 `cuda.py`
那个写法本身就偏离了原意 —— 一旦有人补上生成逻辑,`=0` 会在两处得到相反结果。
删掉后这个矛盾不存在了。

保留 `NVTE_PLUGIN_SKIP_CUDA` 的理由:它能在**有 CUDA 卡**的机器上强制禁用 CUDA 后端
(测试 fallback 用),也避免在非 CUDA 平台白跑一遍 `_load_cuda_libs()` 的
`ctypes.CDLL` 全局加载。不设它时,可用性完全由
`torch.cuda.is_available()` + `import transformer_engine_torch_nv` 探测决定。

**下游待办(本次未动)**:`FlagScale` 和 `Megatron-LM-FL` 里约 45 处 `TE_FL_SKIP_CUDA=1`
(8 个厂商的 `.github/configs/*.yml`、`tools/install/*/env.sh`、`set_env_*.sh`,
以及 `prepare_dependencies.yml:167` / `prepare_te_fl.yml:104` 两处 "python-only 构建必须
`TE_FL_SKIP_CUDA=1`" 的 workflow 门控)。这些目前消费的是**旧仓
`flagos-ai/TransformerEngine-FL`** —— 两个下游仓共 24 处引用都指向旧仓,零处引用本仓,
而旧仓的 `common/__init__.py:21-38` `skip_cuda_build()` 只认 `TE_FL_SKIP_CUDA`。
所以现在改名会直接打断 8 个平台的 CI,必须等下游切到新 fork 时一起改。

### 5.2 `TE_FL_STRICT` 语义不一致已修复

**语义确定为:设置了就不允许 fallback。** 原先 `policy.py` 和 `manager.py` 读同一个变量
却当成两件相反的事,默认值也不同:

| | `policy.py:252` | `manager.py:386`(旧) |
| --- | --- | --- |
| 读法 | `get("TE_FL_STRICT", "0") == "1"` | `getenv("TE_FL_STRICT", "1") != "0"` |
| 含义 | `strict`(严格模式) | `enable_fallback`(开启回退) |

```
修复前:
TE_FL_STRICT 未设置 -> policy.strict=False  enable_fallback=True   # 合理
TE_FL_STRICT=1      -> policy.strict=True   enable_fallback=True   # 要严格,却开着回退
TE_FL_STRICT=0      -> policy.strict=False  enable_fallback=False  # 不要严格,却关了回退
```

修复方式:`manager.call()` 不再自己读环境变量,改为 `enable_fallback = not get_policy().strict`。
`policy.py:252` 成为该变量的**唯一读取点**,两者不可能再分叉。实测:

```
TE_FL_STRICT 未设置 -> strict=False -> 主实现失败后回退到下一候选
TE_FL_STRICT=0      -> strict=False -> 同上
TE_FL_STRICT=1      -> strict=True  -> 主实现的异常直接抛出,不回退
```

顺带修好的两点:

- **`policy_context()` / `with_strict_mode()` 现在真的能控制 fallback**。旧实现里
  `call()` 只看环境变量,所以这两个 context manager 对回退行为完全没有影响 ——
  `with_strict_mode()` 内部依然会回退
- `resolve()` 里 `chosen is None` 的两个分支原本只有报错信息不同、行为完全一样,
  而 "under strict policy" 的措辞会让人以为 `strict` 管的是**选择**失败。实际上这里
  是"没有任何候选匹配 order token",与 `strict` 无关(两种情况都无可回退),
  因此合并为信息更全的那条

**对下游的影响**:`FlagScale` / `Megatron-LM-FL` 当前没有任何地方设置 `TE_FL_STRICT`
(已 grep 确认),所以这次修复不改变现有作业的行为。

## 6. 环境

```bash
conda activate /share/project/lixianduo/envs/fl-train-base   # python 3.12, torch 2.7.1+cu128
export NVTE_PLUGIN=transformer_engine_plugin_fl
```

常用环境变量(选择策略类仍是 `TE_FL_*` 前缀,见 §5):

| 变量 | 作用 |
| --- | --- |
| `NVTE_PLUGIN` | 设为 `transformer_engine_plugin_fl` 启用插件 |
| `NVTE_PLUGIN_SKIP_CUDA` | `1` 跳过 CUDA:构建期不编译扩展、加载期不探测 `transformer_engine_torch_nv`。唯一的 skip-cuda 开关,接口仓与插件共用(见 §5.1) |
| `TE_FL_PREFER` | 优先后端:`flagos` / `vendor` / `reference` |
| `TE_FL_PER_OP` | 按 op 指定,如 `layernorm_fwd=vendor\|flagos;rmsnorm_fwd=flagos` |
| `TE_FL_ALLOW_VENDORS` / `TE_FL_DENY_VENDORS` | 厂商白/黑名单(逗号分隔) |
| `TE_FL_STRICT` | `1` 禁止 fallback:主实现失败时直接抛异常,不再尝试下一候选(默认不设 = 允许回退,见 §5.2) |

`TE_FL_PER_OP` 的格式是 `op=后端1|后端2;op2=后端3`,用 `;` 分隔 op、`|` 分隔候选顺序。
