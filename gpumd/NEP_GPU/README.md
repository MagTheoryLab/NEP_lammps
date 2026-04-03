NEP_GPU：GPUMD NEP 的 LAMMPS GPU 接口
====================================

`NEP_GPU` 目录提供：

- NEP 的 CUDA 后端库：`libnep_gpu`（基于 GPUMD NEP 内核）
- LAMMPS 扩展包：`USER-NEP-GPU`

支持的 LAMMPS pair_style：

- `pair_style nep/gpu`（经典非 Kokkos）
- `pair_style nep/gpu/kk`（Kokkos + CUDA）
- `pair_style nep/spin/gpu`（经典非 Kokkos，自旋；需要 `atom_style spin`）
- `pair_style nep/spin/gpu/kk`（Kokkos + CUDA，自旋；需要 `atom_style spin`）

---

一、在 GPUMD 中构建 `libnep_gpu`
-----------------------------------

### 1.1 推荐：CMake（静态库）

```bash
cd /path/to/GPUMD
cmake -S NEP_GPU -B build-nep_gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=61   # 按你的 GPU 改：61/75/80/86/90...
cmake --build build-nep_gpu -j
```

产物：

```text
/path/to/GPUMD/build-nep_gpu/libnep_gpu.a
```

### 1.2 可选：CMake（共享库，Makefile 链接更省事）

```bash
cd /path/to/GPUMD
cmake -S NEP_GPU -B build-nep_gpu-so \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEP_GPU_BUILD_SHARED=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build-nep_gpu-so -j
```

产物：

```text
/path/to/GPUMD/build-nep_gpu-so/libnep_gpu.so
```

---

二、在 LAMMPS 中使用（Makefile 方式：适合较老版本/无 CMake）
------------------------------------------------------

适用场景：LAMMPS 比较老、或者你的环境里 CMake/Kokkos/CUDA 组合不容易配通。

### 2.1 安装 `USER-NEP-GPU` 包源码

把包目录拷贝到 LAMMPS 源码树（目录名必须是 `USER-NEP-GPU`）：

```bash
cp -r /path/to/GPUMD/NEP_GPU/interface/lammps/USER-NEP-GPU /path/to/lammps/src/USER-NEP-GPU
```

在 LAMMPS `src/` 下启用包：

```bash
cd /path/to/lammps/src
make yes-user-nep-gpu
```

如果要用自旋版，还需要启用 LAMMPS 的 `SPIN` 包（提供 `atom_style spin`）：

```bash
make yes-spin
```

如果要用 `/kk`（Kokkos 版），还需要启用 `KOKKOS` 包并选择带 CUDA 的 Kokkos 编译选项：

```bash
make yes-kokkos
```

### 2.2 修改 machine Makefile：加入头文件与库

以 `src/MAKE/Makefile.mpi`（或 `Makefile.serial`）为例，加入以下内容（路径按实际修改）：

```make
GPUMD = /path/to/GPUMD

# 让 LAMMPS 能找到 nep_gpu_model_*.cuh
LMP_INC += -I$(GPUMD)/NEP_GPU/src -I$(GPUMD)/src -I$(GPUMD)/src/force

# 链接 NEP GPU 后端（静态库示例；注意 CUDA 相关库放在 libnep_gpu.a 后面）
NEP_GPU_LIB = $(GPUMD)/build-nep_gpu/libnep_gpu.a
LIB += $(NEP_GPU_LIB) -lcudart -lcublas -lcusolver -lcufft
```

如果你使用共享库（`libnep_gpu.so`），通常可以：

```make
NEP_GPU_SO = $(GPUMD)/build-nep_gpu-so
LIB += -L$(NEP_GPU_SO) -lnep_gpu
```

提示：

- 如果出现 `cannot find -lcublas` 之类错误，给 `LIB` 补上 CUDA 库路径（例如 `-L${CUDA_HOME}/lib64`），或设置 `LD_LIBRARY_PATH`。
- Makefile 模式下，`USER-NEP-GPU` 的 `.cpp/.h` 会被安装到 LAMMPS `src/` 目录；但 `.cuh` 仍在 GPUMD 里，因此需要上面的 `-I$(GPUMD)/NEP_GPU/src`。

### 2.3 编译 LAMMPS

```bash
cd /path/to/lammps/src
make mpi -j
```

---

三、在 LAMMPS 中使用（CMake 方式：推荐）
-----------------------------------

说明：`USER-NEP-GPU` 不是 LAMMPS 上游自带包，需要把包源码和 CMake 模块拷进 LAMMPS 源码树，并在 `cmake/CMakeLists.txt` 注册。

本仓库提供一键脚本：`NEP_GPU/interface/lammps/cmake/enable_user-nep-gpu_cmake.sh`。

```bash
/path/to/GPUMD/NEP_GPU/interface/lammps/cmake/enable_user-nep-gpu_cmake.sh \
  /path/to/lammps \
  /path/to/GPUMD
```

Kokkos + CUDA（示例）：

```bash
cd /path/to/lammps
cmake -S cmake -B build-kokkos-nep \
  -DCMAKE_BUILD_TYPE=Release \
  -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON \
  -DPKG_USER-NEP-GPU=ON \
  -DNEP_GPU_SOURCE_DIR=/path/to/GPUMD \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DFFT_KOKKOS=CUFFT
cmake --build build-kokkos-nep -j
```

自旋版额外要求：

- CMake：加 `-DPKG_SPIN=ON`
- 输入脚本：用 `atom_style spin`

---

四、运行时注意事项
--------------------

1. **`run.in` 依赖（DFTD3）**

`src/force/nep.cu` 中的 `NEP::initialize_dftd3()` 会尝试读取当前工作目录下的 `run.in`，如果找不到，会报错：

```text
Error text: Cannot open run.in.
```

如果你不使用 DFTD3 校正，一个简单做法是在运行目录下放一个空的 dummy `run.in`：

```bash
echo "# dummy run.in for NEP_GPU" > run.in
```

2. **多 GPU / MPI 的 device 选择**

- `pair_style nep/gpu`：`NEP_GPU_DEVICE=<id>`（相对 `CUDA_VISIBLE_DEVICES` 的索引）
- `pair_style nep/spin/gpu`：`NEP_SPIN_GPU_DEVICE=<id>`（或回退 `NEP_GPU_DEVICE`）
- `/kk` 版本通常由 Kokkos 管理设备，推荐直接用 `-k on g <ngpu>`

3. **`/kk` 运行开关（Kokkos）**

- 需要启用 Kokkos：`-k on g 1 -sf kk -pk kokkos neigh full newton off`
- 输入脚本用对应样式：`atom_style atomic/kk`、`run_style verlet/kk`

4. **常用排错/性能环境变量**

- `NEP_GPU_LMP_DEBUG=1`：输出更多诊断信息
- `NEP_GPU_LMP_VALIDATE_NL=1`：进入 NEP kernel 前校验 NN/NL（越界会直接报错）
- `NEP_GPU_LMP_CACHE_NL=1`：仅在 LAMMPS 重建 neighbor list 时打包 NL/NN，中间步复用
- `NEP_GPU_LMP_FORCE_FP32=1`：force 累积用 float32（可能有精度损失，但可提升部分老卡性能）

---

五、简单 LAMMPS 示例
---------------------

非自旋：

```lammps
units           metal
boundary        p p p
atom_style      atomic

read_data       C.bilayer.data

pair_style      nep/gpu
pair_coeff      * * nep.C.GAP2020 C C

thermo_style    custom step pe temp press
thermo          100

timestep        0.001
velocity        all create 300 12345
fix             1 all nvt temp 300 300 0.1

run             1000
```

自旋（示意；势文件与元素映射按你的模型填写）：

```lammps
units           metal
boundary        p p p
atom_style      spin
newton          pair on

read_data       spin.data

pair_style      nep/spin/gpu fm_units frequency
pair_coeff      * * nep.spin.model Fe

run             1000
```

---

六、性能 bench（可选：模拟 LAMMPS 调用）
-----------------------------------

源码：`tools/bench_nep_lmp_call.cu`

### 编译（从仓库根目录）

```bash
nvcc -std=c++17 -O3 -arch=sm_60 -I./src -I./src/force -I./NEP_GPU/src \
  tools/bench_nep_lmp_call.cu \
  NEP_GPU/src/nep_gpu_model_lmp.cu NEP_GPU/src/nep_lmp_bridge.cu \
  src/model/box.cu src/force/potential.cu src/utilities/error.cu src/utilities/read_file.cu \
  -o bench_nep_lmp_call
```

> 请按你的 GPU 架构调整 `-arch=sm_60`（例如 `sm_75`/`sm_80`）。

### 运行示例

```bash
./bench_nep_lmp_call potentials/nep/C_2024_NEP4.txt --natoms 20000 --density 0.02 --steps 200 --warmup 20 --neigh-every 10
```
