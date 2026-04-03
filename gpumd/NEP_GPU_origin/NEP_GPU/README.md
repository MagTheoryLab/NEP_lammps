NEP_GPU：GPUMD NEP 的 LAMMPS GPU 接口
====================================

`NEP_GPU` 目录提供了：

- 基于 GPUMD 内核（`src/force/nep.cu` 等）的 GPU NEP 封装（`NepGpuModel`）；
- 一套 LAMMPS 扩展包 `USER-NEP-GPU`，在 LAMMPS 中提供 `pair_style nep/gpu`。

设计思路：

- 所有 CUDA kernel 和 NEP 逻辑仍然在 GPUMD 仓库内部；
- LAMMPS 只通过一个很薄的 C++ 接口（`NepGpuModel` + `pair_nep_gpu`）访问；
- 用户手动在 GPUMD 里编译出 `libnep_gpu.a` 静态库，然后在构建 LAMMPS 时链接进去。

下面分别说明：

1. 在 GPUMD 中构建 `libnep_gpu.a`
2. 在 LAMMPS 中使用（Makefile 方式）
3. 在 LAMMPS 中使用（CMake 方式）
4. 运行时注意事项
5. 简单 LAMMPS 示例

---



一、在 GPUMD 中构建 `libnep_gpu.a`
-----------------------------------

假设当前目录是 GPUMD 仓库根目录（也就是有 `src/`、`NEP_GPU/` 等子目录的地方）。

1. 清理当前目录下旧的 `.o` 和库（可选）

```bash
cd /path/to/GPUMD
rm -f *.o libnep_gpu.a
```
2. 修改下src/force/nep.cuh 增加两个函数

```c++
  // Lightweight accessors for external interfaces (e.g. NepGpuModel).
  float get_rc_radial() const { return paramb.rc_radial; }
  int get_num_types() const { return paramb.num_types; }
```


3. 用 `nvcc` 编译 GPU 相关源码

```bash
nvcc -std=c++14 -O3 -arch=sm_60 -I./src \
  -c \
  src/model/box.cu \
  src/force/potential.cu \
  src/force/nep.cu \
  src/force/neighbor.cu \
  src/force/dftd3.cu \
  src/utilities/error.cu \
  src/utilities/read_file.cu \
  NEP_GPU/src/nep_gpu_model.cu
```

- 请根据自己的 GPU 架构修改 `-arch=sm_60`（例如 `sm_70`、`sm_80` 等）；
- 如果算力 < 6.0，可能需要像 `src/makefile` 里那样加上 `-DUSE_KEPLER`。

4. 打包成静态库

```bash
ar rcs libnep_gpu.a *.o
```

构建完成后，你应该在 GPUMD 根目录看到：

```text
/path/to/GPUMD/libnep_gpu.a
```

---

二、在 LAMMPS 中使用（Makefile 方式）
--------------------------------------

以下假设你的 LAMMPS 源码在：

```text
/opt/software/lammps-29Aug2024/src
```

### 2.1 拷贝 USER-NEP-GPU 包

```bash
cd /opt/software/lammps-29Aug2024/src

# 把 USER-NEP-GPU 目录拷到 LAMMPS src
cp -r /path/to/GPUMD/NEP_GPU/interface/lammps/USER-NEP-GPU ./USER-NEP-GPU
```

### 2.2 拷贝 GPU NEP 静态库

最简单的方式是把 `libnep_gpu.a` 直接放到 LAMMPS `src/`：

```bash
cp /path/to/GPUMD/libnep_gpu.a .
```

### 2.3 启用 USER-NEP-GPU 包

在 LAMMPS `src` 目录执行：

```bash
make yes-user-nep-gpu
```

这会调用 `USER-NEP-GPU/Install.sh`，把以下文件复制到 `src/`：

- `pair_nep_gpu.cpp`
- `pair_nep_gpu.h`
- `nep_gpu_model.cuh`（NEP_GPU 对外接口头文件）

> 说明：当前 `Install.sh` 会尝试在 `Makefile.package` 中追加 / 清理 NEP_GPU 相关的 `PKG_SYSLIB` / `PKG_SYSPATH` 设置，避免重复追加同一个 `-L` 或 `-l`。如果你只想用完全手工方式维护编译选项，可以自行编辑 `Makefile.package.settings` 并移除这部分自动修改。

### 2.4 在 `Makefile.package.settings` 中加入链接选项（推荐按一次性方式处理）

为了让配置更清晰、一次性，就建议在 `src/Makefile.package.settings` 中显式写出 NEP_GPU 的链接需求，例如：

```make
PKG_SYSLIB  += -lnep_gpu -lcudart -lcublas -lcusolver -lcufft
PKG_SYSPATH += -L.. -L/opt/devtools/cuda/cuda-12.8.0/lib64
```

这告诉 LAMMPS：当 `USER-NEP-GPU` 包被启用时，在链接阶段需要：

- 链接 `libnep_gpu.a`；
- 同时链接 CUDA 运行时库：`cudart`、`cublas`、`cusolver`、`cufft`；
- 并在 `PKG_SYSPATH` 中显式加入：
  - `-L..`（默认在 LAMMPS `src/` 下找 `libnep_gpu.a`）；
  - `-L/opt/devtools/cuda/cuda-12.8.0/lib64`（请根据你本机 CUDA 安装位置调整，例如 `/usr/local/cuda/lib64`）。

如果你的环境中设置了 `CUDA_HOME` 且其下有 `lib64`，也可以改写为更通用的写法：

```make
PKG_SYSLIB  += -lnep_gpu -lcudart -lcublas -lcusolver -lcufft
PKG_SYSPATH += -L.. -L$(CUDA_HOME)/lib64
```

这样就不需要再在 `Makefile.mpi` 里重复加一遍 `-L/usr/local/cuda/lib64` 之类的选项；而 `Install.sh` 只是在你多次 `make yes-user-nep-gpu` 时，帮你避免 `PKG_SYSLIB/PKG_SYSPATH` 中重复堆叠相同的 `-L/-l`。

### 2.5 （可选）增加 GPUMD 头文件搜索路径

对于 LAMMPS 的 `pair_nep_gpu.cpp` 当前实现来说，`nep_gpu_model.cuh` 已经被复制到了 `src/`，且 `pair_nep_gpu.cpp` 只包含这个头，所以**一般不需要额外给 LAMMPS 加 `-I` 指向 GPUMD 源码目录**。

只有当你想在 `pair_nep_gpu.cpp` 里直接 include GPUMD 的其他头文件（例如调试或扩展）时，才需要在 `Makefile.mpi` 中添加类似：

```make
LMP_INC = -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64 \
          -I/path/to/GPUMD/src \
          -I/path/to/GPUMD/NEP_GPU/src
```

### 2.6 编译 LAMMPS

```bash
cd /opt/software/lammps-29Aug2024/src
make mpi
```

编译成功后，你的 `lmp_mpi` 就支持：

```lammps
pair_style  nep/gpu
pair_coeff  * * nep.C.GAP2020 C C
```

---

三、在 LAMMPS 中使用（CMake 方式）
-----------------------------------

`NEP_GPU/interface/lammps/USER-NEP-GPU.cmake` 已经为 CMake 方式集成了 `pair_nep_gpu`。使用步骤：

1. 拷贝 `USER-NEP-GPU` 到 LAMMPS `src`

   ```bash
   cd /opt/software/lammps-29Aug2024/src
   cp -r /path/to/GPUMD/NEP_GPU/interface/lammps/USER-NEP-GPU ./USER-NEP-GPU
   ```

2. 确保 GPUMD 仓库根目录已构建好 `libnep_gpu.a`（参考第一节）

3. 在 LAMMPS 的 `build` 目录下用 CMake 配置

   ```bash
   cd /opt/software/lammps-29Aug2024/build
   cmake ../cmake \
     -DPKG_USER-NEP-GPU=on \
     -DNEP_GPU_ROOT=/path/to/GPUMD
   make -j
   ```

其中：

- `PKG_USER-NEP-GPU=on`：启用 USER-NEP-GPU 包；
- `NEP_GPU_ROOT`：指向 GPUMD 仓库根目录（即包含 `src/`、`NEP_GPU/` 的目录）。

CMake 脚本会自动：

- 把 `pair_nep_gpu.cpp/.h` 加入 `lammps` 目标；
- 添加头文件搜索路径：
  - `${NEP_GPU_ROOT}/src`
  - `${NEP_GPU_ROOT}/NEP_GPU/src`
- 在 `${NEP_GPU_ROOT}` 和 `${NEP_GPU_ROOT}/lib` 下寻找 `libnep_gpu.a`；
- 将 `libnep_gpu` 以及 CUDA 库（`cudart`、`cublas`、`cusolver`、`cufft`）链接到 `lammps`。

如果 CUDA 安装不在系统默认路径，可以在 CMake 配置时额外指定：

```bash
cmake ../cmake \
  -DPKG_USER-NEP-GPU=on \
  -DNEP_GPU_ROOT=/path/to/GPUMD \
  -DCMAKE_LIBRARY_PATH=/opt/devtools/cuda/cuda-12.8.0/lib64
```

或者在工具链里设置好 `CUDA_HOME` 并通过工具链文件/环境变量导入。

---

四、运行时注意事项
--------------------

1. **`run.in` 依赖（DFTD3）**

`src/force/nep.cu` 中的 `NEP::initialize_dftd3()` 会尝试读取当前工作目录下的 `run.in`，如果找不到，会报错：

```text
Error text: Cannot open run.in.
```

如果你不使用 DFTD3 校正，一个简单的办法是在运行 LAMMPS 的目录下放一个空的 dummy `run.in`，例如：

```bash
echo "# dummy run.in for NEP_GPU" > run.in
```

更彻底的做法是修改 GPUMD 中 `NEP::initialize_dftd3()`，让它在找不到 `run.in` 时直接返回并设定 `has_dftd3 = false`，但这会改动 GPUMD 本身，这里不强制。

2. **盒子映射（支持一般晶格）**

当前的 `PairNEPGPU` 实现中，我们通过 LAMMPS 的 triclinic 盒子参数 `domain->h[0..5]` 构造 GPUMD 的 3×3 box 矩阵（`Box::cpu_h`），映射关系是：

- LAMMPS 内部使用 `h0..h5` 表示：
  - `h[0] = xprd`
  - `h[1] = yprd`
  - `h[2] = zprd`
  - `h[3] = xy`
  - `h[4] = xz`
  - `h[5] = yz`
- 在 `pair_nep_gpu.cpp` 中构造的 GPUMD box（列为晶格向量）为：
  - `a = (xprd, 0,    0   )`
  - `b = (xy,   yprd, 0   )`
  - `c = (xz,   yz,   zprd)`

即：

```c++
// column-major: [a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, c_z]
h[0] = domain->h[0]; // a_x = xprd
h[3] = 0.0;          // a_y
h[6] = 0.0;          // a_z

h[1] = domain->h[3]; // b_x = xy
h[4] = domain->h[1]; // b_y = yprd
h[7] = 0.0;          // b_z

h[2] = domain->h[4]; // c_x = xz
h[5] = domain->h[5]; // c_y = yz
h[8] = domain->h[2]; // c_z = zprd
```

这与 GPUMD 读取 EXYZ `Lattice=` 的内部约定一致，因此：

- 正交盒（`xy=xz=yz=0`）自然支持；
- 一般 triclinic 盒子（带倾斜）也能正确传递到 NEP_GPU。

3. **模型支持范围**

当前 GPU 接口主要针对“标准 NEP”（能量 + 力 + virial）：

- 支持 NEP3 / NEP4 / NEP5 势文件；
- 暂未直接暴露 NEP_Spin、NEP_Charge 等模型；
- 将来如果你在 GPUMD 中把 NEP_Spin/NEP_Charge 的 force 路径也迁移到 `src/force/` 层，可以按相同思路封装一个 `NepGpuModelSpin` 及对应的 LAMMPS pair。

---

五、简单 LAMMPS 示例
---------------------

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

确保：

- `nep.C.GAP2020` 是一个合法的 NEP 势文件（GPUMD/NEP_CPU 能正常读取）；
- 运行目录下存在 `run.in`（即便只是一个 dummy 文件），除非你已经修改了 `NEP::initialize_dftd3()` 以去掉这个硬依赖。
