# GLSD（Ma & Dudarev, PhysRevB.86.054416）在 USER-TSPIN 中的公式推导与代码实现

本文面向“讲清楚推导 + 对上实现”。核心目标：把论文中的 GLSD（generalized Langevin spin dynamics）方程，按本工程的单位与数据结构约定，推到 `fix glsd/*` 的可执行离散形式，并逐段对照到代码（`src/USER-TSPIN/fix_glsd_nh.*`）。

> 本工程的关键约定：把每个原子的磁矩向量 **直接当作 “μB 个数” 的数值** 来存；有效场（或“磁力”）用能量对这个数值磁矩的导数表示，因此 **玻尔磁子常数 μ\_B^(0) 在工程实现里被吸收**，不会在方程中显式出现。

---

## 0. 代码入口与文件结构

实现文件：

- `src/USER-TSPIN/fix_glsd_nh.h` / `src/USER-TSPIN/fix_glsd_nh.cpp`：核心 GLSD 积分器 + 与 Nose–Hoover（FixNH）晶格积分耦合。
- `src/USER-TSPIN/fix_glsd_nvt.h`：`FixStyle(glsd/nvt,FixGLSDNVT);`（在 NVT 外层加 GLSD 自旋演化）。
- `src/USER-TSPIN/fix_glsd_npt.h`：`FixStyle(glsd/npt,FixGLSDNPT);`（在 NPT 外层加 GLSD 自旋演化）。

在 LAMMPS 输入脚本里你真正用的是 `fix ... glsd/nvt ... glsd ...` 或 `fix ... glsd/npt ... glsd ...`；`FixGLSDNH` 是这两个 style 的基类。

---

## 1. 工程单位与变量对应（μB 被吸收的版本）

### 1.1 本工程的单位与含义（NEP(SPIN) + USER-TSPIN 自洽约定）

先强调一句：LAMMPS 的 `atom_style spin` 只是“存数组”，并不会自动给自旋赋予某种物理单位；**单位/物理意义完全由我们自己定义**。本工程采用的约定是自洽的，但不追求与 LAMMPS 其它 SPIN 生态（例如某些 fix 的历史写法）完全兼容。

固定约定（核心）：

- `atom->sp[i][0..2]`：自旋方向（单位向量）
- `atom->sp[i][3]`：自旋幅值，解释为磁矩大小 `|M|`（单位：μB）
- 能量单位：`eV`
- `atom->fm[i][0..2]`：有效场（“磁力”）采用**场模式**写入：  
$$
\mathbf{H} = -\frac{\partial E}{\partial \mathbf{M}}\quad(\text{单位：eV}/\mu_B)
$$

本工程中“场 → 频率”的转换（用于进动）：

$$
\omega = (g/\hbar)\,H
$$
  对应代码 `FixGLSDNH::fm_to_frequency()`（`H` 取自 `atom->fm`）。

当前支持范围（实现侧约束，讲解时建议提前说明）：

- NEP spin pair style（例如 `pair_style nep/spin/gpu`、`pair_style nep/spin/gpu/kk`）：要求能往 `atom->fm` 写入 `H=-dE/dM`（`eV/μB`，场模式）
- USER-TSPIN 的 `fix glsd/*`：要求 `atom->fm` 为上述场模式（文档中例子用 `fm_units field` 只是做显式校验）
- 不提供 `spin_units` 选项：`sp[3]` 永远按 μB 解释
- `fix precession/spin` 不允许与 USER-TSPIN 的 `fix glsd/*` 混用（代码在 `FixGLSDNH::init()` 会直接报错）；外场支持改用 `setforce/spin` 或 `tspin/precession/spin`（`zeeman` 直接输入 `H`，单位 `eV/μB`；Kokkos 建议用 `tspin/precession/spin/kk`），并在内部重算阶段通过白名单重放保持一致

### 1.2 Atom 数据结构（实现侧）

在代码中我们对 LAMMPS `atom_style spin` 的解释如下：

- `atom->sp[i][0..2]`：自旋方向（单位向量）
- `atom->sp[i][3]`：自旋幅值，**解释为磁矩大小 \|M\|（单位：μB）**
- 因而该原子的**磁矩向量**可写成
$$
\mathbf{M}_i = \underbrace{(\text{sp}[3])}_{\mu_B}\cdot\underbrace{\frac{(\text{sp}[0..2])}{\|\text{sp}[0..2]\|}}_{\text{方向}}
$$
- `atom->fm[i][0..2]`：有效场（工程约定为场模式）  
$$
\mathbf{H}_i = -\frac{\partial \mathcal{H}}{\partial \mathbf{M}_i}\quad(\text{单位：eV}/\mu_B)
$$

### 1.3 物理量与工程量的“吸收关系”

用下标 `phys` 表示真实物理量（带 SI 单位），无下标表示工程存储/计算的量：

- 物理磁矩与工程磁矩：
$$
\mathbf{M}_{\rm phys} = \mu_B^{(0)} \mathbf{M}
$$
  其中 $\mathbf{M}$ 在代码里就是“μB 个数”（典型 1–5），因此视作数值向量。
- 物理磁场与工程有效场：
$$
\mathbf{H} = \mu_B^{(0)}\,\mathbf{B}_{\rm eff}
$$
  于是 $\mathbf{H}$ 的量纲是 eV/μB（工程实现里仍以 eV 体系运算）。
- 这会导致原本在一些写法里出现的 $\mu_B^{(0)}$ 在两边链式法则中抵消，最终动力学方程里只显式保留 $\hbar$（eV·time）和 $g$。

---

## 2. 从论文 Eq.(18)(19) 到工程方程（你给的框架 + 必要补充）

下面严格沿论文的数学结构做变量代换，但按工程约定把 $\mu_B^{(0)}$ 吸收掉。

### 2.1 论文起点（单自旋的广义朗之万方程）

论文给出的形式（以 $\mathbf{S}$ 为自旋变量）可概括为：
$$
\frac{d\mathbf{S}}{dt}
=\frac{1}{\hbar}\,\mathbf{S}\times\left(-\frac{\partial \mathcal{H}}{\partial \mathbf{S}}\right)
-\gamma_s'\frac{\partial \mathcal{H}}{\partial \mathbf{S}}
+\boldsymbol{\xi}(t)
$$

其中：

- 第一项：进动（能量守恒部分）
- 第二项：耗散（趋向能量极小）
- 第三项：噪声（热涨落），满足涨落-耗散关系

### 2.2 变量代换：把论文 $\mathbf{S}$ 换成工程的磁矩 $\mathbf{M}$

工程采用（吸收 μB 后的）映射：
$$
\mathbf{S}=-\frac{\mathbf{M}}{g}\quad\Longleftrightarrow\quad \mathbf{M}=-g\mathbf{S}
$$

定义工程有效场：
$$
\mathbf{H}\equiv-\frac{\partial \mathcal{H}}{\partial \mathbf{M}}
$$

链式法则：
$$
\frac{\partial \mathcal{H}}{\partial \mathbf{S}}
=\frac{\partial \mathcal{H}}{\partial \mathbf{M}}\cdot\frac{\partial \mathbf{M}}{\partial \mathbf{S}}
=(-\mathbf{H})\cdot(-g)=g\mathbf{H}
$$

时间导数：
$$
\frac{d\mathbf{S}}{dt}=-\frac{1}{g}\frac{d\mathbf{M}}{dt}
$$

### 2.3 得到工程方程（连续形式）

代回论文方程并两边乘以 $-g$，得到：
$$
\frac{d\mathbf{M}}{dt}
=-\frac{g}{\hbar}\,\mathbf{M}\times\mathbf{H}
\lambda\,\mathbf{H}
\boldsymbol{\eta}(t)
$$
其中
$$
\lambda \equiv \gamma_s' g^2,\qquad \boldsymbol{\eta}\equiv -g\boldsymbol{\xi}
$$

噪声满足（白噪声）：
$$
\langle \eta_\alpha(t)\rangle=0,\qquad
\langle \eta_\alpha(t)\eta_\beta(t')\rangle
=2\lambda k_B T\,\delta_{\alpha\beta}\delta(t-t')
$$

这正是 `fix_glsd_nh.h` 中注释写的实现目标（`dM/dt = -(g/ħ) M×H + λ H + η`）。

---

## 3. “eb2b” 对应的数值积分细节：分裂 + Boris 旋转（代码核心）

### 3.1 把 GLSD 拆成三部分算子 A/B/C

把
$$
\dot{\mathbf{M}}=-(g/\hbar)\mathbf{M}\times\mathbf{H}+\lambda\mathbf{H}+\boldsymbol{\eta}
$$
拆成（在一个小时间步内把 $\mathbf{H}$ 视作常量的意义下）：

- **A（进动）**：
$$
\dot{\mathbf{M}}=\mathbf{M}\times\boldsymbol{\Omega},\qquad \boldsymbol{\Omega}\equiv-(g/\hbar)\mathbf{H}
$$
- **B（耗散漂移）**：
$$
\dot{\mathbf{M}}=\lambda\mathbf{H}
$$
- **C（噪声）**：
$$
\dot{\mathbf{M}}=\boldsymbol{\eta}(t)
$$

`FixGLSDNH::glsd_step()` 与 `FixGLSDNH::glsd_map()` 使用的就是二阶 Strang 分裂：
$$
e^{\Delta t(A+B+C)}\approx
e^{\frac{\Delta t}{2}(B+C)}\;e^{\Delta t A}\;e^{\frac{\Delta t}{2}(B+C)}
$$
对应代码里的 “half-kick 1/2（B+C） + rotation（A） + half-kick 2/2（B+C）”。

### 3.2 B+C 半步：漂移 + 噪声的离散（与涨落-耗散严格一致）

对 B 项的显式更新就是：
$$
\mathbf{M}\leftarrow \mathbf{M}+\frac{\Delta t}{2}\lambda\mathbf{H}
$$

对 C 项，使用白噪声离散：令三个分量的随机增量满足
$$
\Delta \mathbf{W}\sim\mathcal{N}(\mathbf{0},\Delta t\,\mathbf{I})
$$
则
$$
\Delta\mathbf{M}_{\rm noise}=\sqrt{2\lambda k_B T}\;\Delta\mathbf{W}
$$
于是整步（时间步 $\Delta t$）的噪声方差为 $2\lambda k_BT\Delta t$，对应连续涨落-耗散关系。

代码实现（`glsd_step`/`glsd_map`）的写法等价于：

- 先计算 $k_BT = \text{force->boltz}\cdot T$
- 设 $\mu_s = 2\lambda k_BT$
- 对一次 `glsd_step(dt_step, ...)`（注意这里的 `dt_step` 可能是 `dt/2`）：
  - 每个 half-kick 的噪声标准差取
$$
\sigma_{\text{kick}}=\sqrt{\frac12\mu_s\,dt_{\rm step}}=\sqrt{\lambda k_B T\,dt_{\rm step}}
$$
  - 同一次 `glsd_step` 内有两次独立 half-kick，因此该 `dt_step` 对应的总噪声方差是 $2\lambda k_BT\,dt_{\rm step}$
  - 在显式预测-校正模式下：`initial_integrate()` 和 `post_force()` 各调用一次 `glsd_step(dt/2, ...)`，两次相加给出整步方差 $2\lambda k_BT\,dt$

### 3.3 A 步：进动项的“Boris t/s 形式”旋转（这就是你说的 eb2b）

对常 $\boldsymbol{\Omega}$ 的方程
$$
\dot{\mathbf{M}}=\mathbf{M}\times\boldsymbol{\Omega}
$$
精确解是一段绕 $\boldsymbol{\Omega}$ 方向的刚体旋转（保持 $|\mathbf{M}|$ 不变）。

代码没有直接构造旋转矩阵，而是使用 Boris 积分常用的 t/s（Cayley 变换）形式，数值上：

- 定义 $\mathbf{t}=\frac{\Delta t}{2}\boldsymbol{\Omega}$, $t^2=\|\mathbf{t}\|^2$
- 定义 $\mathbf{s}=\frac{2\mathbf{t}}{1+t^2}$
- 两次叉乘完成旋转：
$$
\mathbf{v}'=\mathbf{M}+\mathbf{M}\times\mathbf{t},\qquad
\mathbf{M}\leftarrow \mathbf{M}+\mathbf{v}'\times\mathbf{s}
$$

这部分对应 `glsd_step()` / `glsd_map()` 中的：

- `Hx_w = fm_to_frequency(fm_x)`：把 $H_x$ 转成 $(g/\hbar)H_x$
- `Ox = -Hx_w`：构造 $\Omega_x = -(g/\hbar)H_x$
- `tx = 0.5*dt*Ox`：构造 $\mathbf{t}$
- `sx = 2*tx/(1+t2)`：构造 $\mathbf{s}$
- 两次叉乘更新 `Sx,Sy,Sz`

该旋转步的好处：在大进动频率（THz）下仍稳定，且（在只有 A 项时）严格保持自旋长度。

---

## 4. 与晶格（Nose–Hoover）耦合：整体时间推进顺序

`FixGLSDNH` 继承自 `FixNH`，因此晶格部分仍是标准的 NVT/NPT（Nosé–Hoover 链）推进；GLSD 自旋演化被嵌入到这个时间框架里，实现二阶耦合。

### 4.1 隐式中点（默认 `midpoint_iter = 3`）

当前实现 **强制使用隐式中点自洽**：`midpoint_iter` 必须 $\ge 2$（默认 3）。旧的 `midpoint_iter=1` 显式预测-校正路径已移除。

- `setup()`：缓存上一时刻场到 `glsd_fm_cache`
- `initial_integrate()`：保存起点自旋 $\\mathbf{M}^n$（`lattice on` 用 `s0_cache`），推进晶格到 $x^{n+1}$
- 常规力场评估：在 $(x^{n+1}, \\mathbf{M}^n)$ 得到初始猜测场
- `post_force()`：调用 `solve_spin_midpoint()` 做中点自洽并在 $(x^{n+1}, \\mathbf{M}^{n+1})$ 重算力/场，缓存 $\\mathbf{H}^{n+1}$

> 因为内部会重算力/场，对外场 fix 做了白名单重放：`setforce/spin` / `tspin/precession/spin` 会在 `recompute_force_and_field()` 后通过 `replay_external_spin_fields()` 再加回去，避免外场丢失。

对 `glsd/*/kk`，中点求解后端是自动选择的：

- 默认优先 `DEVICE`（GPU）后端执行隐式中点迭代。
- 若检测到 host 风格外场重放 fix（例如 `setforce/spin` 非 `/kk` 或 `/kk/host`），自动回退到 `HOST` 后端。
- 若当前 execution space 不是 device，或 device scratch 不可用，也会自动回退到 `HOST`。
- 建议外场优先使用 `tspin/precession/spin/kk`（或 `/kk/device`）以保持 GPU 路径。

### 4.2 隐式中点（`midpoint_iter >= 2`）：自洽的 $\mathbf{H}(\mathbf{M}^{n+1/2})$

当 $\mathbf{H}$ 对 $\mathbf{M}$ 的依赖很强时，需要中点自洽来改善稳定性与漂移问题；本工程默认启用该路径。

目标方程（固定点形式）：
$$
\mathbf{M}^{n+1}=\Phi\!\left(\Delta t,\mathbf{M}^n,\mathbf{H}\!\left(\frac{\mathbf{M}^n+\mathbf{M}^{n+1}}{2}\right)\right)
$$
其中 $\Phi$ 就是 `glsd_map(dt, s0, fm(mid), noise_phase, s_out)` 代表的“一步映射”（内部仍是 half(B+C)+A+half(B+C)）。

实现位置：`FixGLSDNH::solve_spin_midpoint()`：

- 保存起点 $\mathbf{M}^n$ 到 `s0_cache`（晶格移动模式）或 `s0`（晶格冻结模式）
- 迭代：
  1. 构造中点自旋 $ \mathbf{M}^{n+1/2} = 0.5(\mathbf{M}^n+\mathbf{M}^{n+1,\text{guess}})$
  2. 用该中点自旋重算一次场 `recompute_force_and_field()`
  3. 用 `glsd_map()` 得到新的 $\mathbf{M}^{n+1}$ 映射值
  4. 继续迭代直到次数到达 `midpoint_iter` 或满足 `midpoint_tol`

注意：由于中点法会在一个时间步内触发多次力/场重算，实现中要求 `fix glsd/*` 必须是最后一个 time_integrate fix（见 `FixGLSDNH::init()` 的检查）。

---

## 5. 参数与代码变量对照表

| 数学符号 | 含义 | 代码位置/变量 |
|---|---|---|
| $\mathbf{M}$ | 磁矩向量（μB 个数） | `atom->sp` 组合出的 `(Sx,Sy,Sz)` |
| $\mathbf{H}=-\partial\mathcal{H}/\partial\mathbf{M}$ | 有效场（eV/μB） | `atom->fm` 或 `fm_use` |
| $\hbar$ | 约化普朗克常数（eV·time） | `hbar = force->hplanck/(2π)` |
| $g$ | 朗德因子（这里写死为 2） | `constexpr double g=2.0` |
| $g/\hbar$ | 频率系数 | `g_over_hbar`，`fm_to_frequency()` |
| $\lambda$ | GLSD 耗散系数 | `lambda`（输入关键字 `glsd alpha` 或 `glsd lambda`；`glsd gammas` 为兼容别名） |
| $T$ | 自旋温度（K） | `spin_temperature`（`glsd stemp`） |
| $k_B$ | 玻尔兹曼常数（eV/K） | `force->boltz` |
| $\eta$ | 噪声（白噪声） | `gaussian_u64()` 生成的高斯增量 |
| $\Delta t$ | 时间步长 | `update->dt` |

---

## 6. 输入脚本关键用法（示例）

以 `glsd/nvt` 为例（`FixNH` 的 `temp ...` 参数仍然要写在前面，GLSD 扩展块从关键字 `glsd` 开始）：

```lammps
units           metal
atom_style      spin

# ... pair_style / pair_coeff 需要能写 atom->fm (H = -dE/dM)

fix 1 all glsd/nvt temp 300 300 0.1 glsd \
  gammas 1.0e-3 stemp 300 seed 12345 fm_units field \
  lattice on midpoint_iter 3
```

常用选项说明（都在 `glsd` 扩展块内）：

- `lambda <λ>`：直接指定本文的 $\lambda$（内部耗散/噪声系数），必须 $\ge 0$（高级选项；一般不推荐手动换算）
- `gammas <λ>`：`lambda` 的兼容别名（历史遗留；不建议继续使用）
- `alpha <α>`：推荐用法。指定无量纲阻尼强度 $\alpha$，代码内部自动换算  
$$
\lambda = \alpha\,(g/\hbar)
$$
  不能与 `gammas` 同时出现
- `stemp <Ts>`：
  - `>0`：自旋噪声温度 $T_s$
  - `0`：跟随晶格温度 compute（`temperature->compute_scalar()`）
  - `-1`：关闭噪声（等价于 $T=0$，但若 `gammas>0` 仍保留耗散漂移）
- `seed <int>`：噪声种子（正整数）
- `lattice on/off`：是否推进晶格（off 时 barostat 失效，只演化自旋）
- `midpoint_iter <n>`：隐式中点迭代次数（必须 `>= 2`）
- `midpoint_tol <eps>`：相对收敛阈值（`0` 表示不提前停止）
- `debug on` / `debug_file ...`：输出能量漂移诊断（实现会记录 `pe_mid`、`pe_end`、`midpoint_backend` 与回退原因）

---

## 7. 为什么你会觉得 `λ=1` “太小”：数量级与推荐选参

你观察到的现象是对的：在本工程的实现里，进动项天然带有 $1/\hbar$ 的大系数，而耗散/噪声项由 $\lambda$ 控制；如果你把 $\lambda$ 当成“和 1 同量级的无量纲数”，那耗散与噪声会小到几乎看不见。

### 7.1 两个“速率尺度”的对比

在代码里（`FixGLSDNH::fm_to_frequency()`）进动角频率来自：
$$
\omega \sim \frac{g}{\hbar} H
$$
而耗散漂移的典型变化率来自：
$$
\left\|\frac{d\mathbf{M}}{dt}\right\|_{\rm diss}\sim \lambda H
$$

因此两者的相对大小主要由
$$
\frac{\lambda}{g/\hbar}=\lambda\frac{\hbar}{g}
$$
决定。

以 `units metal` 为例，$\hbar \approx 6.582\times10^{-16}\,{\rm eV\cdot s}$，$g\approx2$，所以
$$
\frac{g}{\hbar}\approx 3.0\times 10^{15}\ {\rm (eV\cdot s)^{-1}}
$$
这意味着：

- 如果你取 $\lambda=1\ {\rm (eV\cdot s)^{-1}}$，则 $\lambda/(g/\hbar)\sim 3\times10^{-16}$：耗散远小于进动（几乎为零）。
- 如果你希望耗散在“LLG 的 Gilbert 阻尼系数 $\alpha$”那样的相对量级（例如 $\alpha\sim10^{-2}$），一个自然的选择是：
$$
\lambda \approx \alpha\frac{g}{\hbar}\ \Rightarrow\ \lambda \sim 3\times10^{13}\ {\rm (eV\cdot s)^{-1}}
$$

因此更合理的接口是让用户直接给 $\alpha$，由代码内部用当前单位系统的 $\hbar$ 自动换算 $\lambda$；这就是 `glsd alpha` 选项的设计目的。

### 7.2 为什么噪声也会“太小”

涨落-耗散关系是
$$
\langle \eta_\alpha(t)\eta_\beta(t')\rangle = 2\lambda k_B T\,\delta_{\alpha\beta}\delta(t-t')
$$
离散到时间步 $\Delta t$ 时，噪声增量的典型尺度满足
$$
\Delta M_{\rm noise,rms} \sim \sqrt{2\lambda k_B T\,\Delta t}
$$
所以噪声幅度 $\propto \sqrt{\lambda}$。当你把 $\lambda$ 取成 1 这种“无量纲直觉值”时，噪声会小到 $10^{-8}$ 量级（取决于 $\Delta t$、$T$），看起来就像“没有热涨落”。

> 结论：如果你把进动保持为真实的 THz 量级（由 $g/\hbar$ 决定），那么想要“看得见”的耗散/噪声，$\lambda$ 必须是一个带 $1/\hbar$ 尺度的“大数”，而不是 1。

## 8. 常见一致性检查（讲解时很有用）

1) **量纲自洽**  
$\mathbf{H}$ 用 eV/μB、$\hbar$ 用 eV·time，则 $(g/\hbar)\mathbf{H}$ 是 1/time，进动频率量纲正确。

2) **噪声强度是否对上 $2\lambda k_BT$**  
代码用两次 half-kick、每次方差 $\lambda k_BT\Delta t$（或半步时按半步 $\Delta t$ 计算），合起来正好是 $2\lambda k_BT\Delta t$。

3) **只开进动（$\lambda=0$ 且 $T=0$）时是否保长度**  
Boris 旋转步保证数值上几乎严格保持 $|\mathbf{M}|$（误差主要来自浮点舍入）。

4) **可重复性**  
噪声用 `(seed, atom tag, timestep, phase, component)` 生成高斯数，因此并行分解/原子迁移后仍可复现（前提：`tag` 稳定）。

---

## 9. Restart 语义（2026-02 USER-TSPIN 补全后）

### 9.1 新格式 restart（推荐）

- `fix glsd/nh` 与 `fix glsd/nh/kk` 现在写入版本化全局 payload（含 marker+version），并保留 `FixNH` 原有温压链状态。
- 额外保存 GLSD 运行时控制状态：`lattice/midpoint_iter/midpoint_tol/lambda/alpha/stemp/seed`，以及能量诊断连续性所需标量（如 `pe_prev_end`）。
- 每原子缓存 `fm_cache` 与 `s0_cache` 通过 per-atom restart 显式保存，不再依赖“重启后偶然重建”。
- Kokkos 路径与 CPU 路径保持同语义：读入后先落 host，再按 Kokkos dual-view 规则标记/同步。

### 9.2 旧格式 restart（兼容）

- 若读取到无 USER-TSPIN marker 的 legacy restart，仍可继续运行：
  - `FixNH` 基础状态按旧格式恢复；
  - GLSD 专有状态与缓存进入兼容回退路径（保守重建）。
- 每个受影响 fix 实例只会给出一次 warning（包含 fix ID/style），明确提示“不是严格连续重启”。

### 9.3 连续性保证边界

- 在同 fix ID/style、同物理模型、同输入参数下，新格式 restart 目标是“尽可能严格连续”。
- 不承诺跨不同 MPI 划分、不同硬件后端的 bitwise 完全一致；但 restart 状态连续性已最大化。
