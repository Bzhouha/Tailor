# Tailor

Tailor 是一个面向可压缩流动的并行 BiGlobal 线性稳定性求解器。项目使用 Python
完成周期网格和 FD-q 数据准备，使用 C++、PETSc 与 SLEPc 装配并求解广义非厄米
特征值问题。

流向采用

\[
\hat q(x,y,z,t)=q(y,z)e^{i(\alpha x-\omega t)}
\]

并将问题写为

\[
A_{\mathrm{bc}}q=\lambda B_{\mathrm{bc}}q,\qquad
\lambda=-i\omega,\qquad A_{\mathrm{bc}}=-L_{\mathrm{bc}}.
\]

因此 \(\operatorname{Re}(\lambda)>0\) 表示时间增长，求解后通过
\(\omega=i\lambda\) 恢复复频率。

## 功能

- 有界壁法向 \(\xi\) 与周期展向 \(\eta\) 的 FD-q 离散；
- 周期 quintic B-spline 网格及基本流插值；
- PETSc HDF5 缓存校验、自动重建和原子替换；
- 曲线坐标一、二阶 metrics；
- 基本流 \(y,z,yy,zz,yz\) 导数；
- 可压缩线性 Navier--Stokes 点系数和流向 Fourier 变换；
- 基于 `MATBAIJ` 的分布式全局质量矩阵及空间矩阵；
- 等温无滑移壁面和局部特征远场边界；
- SLEPc Krylov--Schur、shift-and-invert 和稀疏 LU；
- 自包含 HDF5 特征值、残差和特征模态输出。

## 依赖

### C++ 与并行数值库

- CMake 3.24 或更高版本；
- 支持 C++20 的编译器；
- MPI；
- complex-scalar PETSc；
- 与 PETSc 兼容的 SLEPc 3.25 或更高版本；
- PETSc 使用的 HDF5、BLAS 和 LAPACK；
- yaml-cpp；
- MUMPS，以及 MUMPS 所需的 ScaLAPACK、METIS/ParMETIS 等并行依赖。

PETSc 必须使用复数标量构建，并建议在配置 PETSc 时直接启用 HDF5 和 MUMPS。
一个典型配置形式如下，具体编译器和 MPI 选项应按计算平台调整：

```bash
./configure \
    --with-scalar-type=complex \
    --download-hdf5 \
    --download-mumps \
    --download-scalapack \
    --download-metis \
    --download-parmetis \
    --download-ptscotch
make all
make check
```

安装说明可参考
[PETSc installation](https://petsc.org/release/install/) 和
[SLEPc installation](https://slepc.upv.es/release/install/install.html)。

### Python

Tailor.py 需要 Python 3.10 或更高版本，以及：

- NumPy；
- SciPy；
- h5py；
- PyYAML；
- petsc4py；
- slepc4py。

`petsc4py` 和 `slepc4py` 必须与 C++ 程序使用同一套 PETSc/SLEPc 和 MPI。对于已有
PETSc/SLEPc 安装，可先设置标准环境变量，再安装 Python bindings：

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-complex
export SLEPC_DIR=/path/to/slepc

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy h5py pyyaml petsc4py slepc4py
```

对于 prefix 安装，可省略 `PETSC_ARCH`。

### 测试附加依赖

完整 CTest 使用 `gfortran` 从参考 Fortran 公式生成固定测试数据。仅构建求解器时，
可以使用 `-DBUILD_TESTING=OFF`，此时不需要 Fortran 编译器。

## 构建

先设置 PETSc 和 SLEPc 环境。CMake 会优先读取 `TAILOR_PETSC_PREFIX`，
否则使用 `PETSC_DIR` 和可选的 `PETSC_ARCH`；SLEPc 类似地从
`TAILOR_SLEPC_PREFIX` 或 `SLEPC_DIR` 查找。

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-complex
export SLEPC_DIR=/path/to/slepc

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

非标准安装位置可以显式指定：

```bash
cmake -S . -B build \
    -DTAILOR_PETSC_PREFIX=/path/to/petsc-prefix \
    -DTAILOR_SLEPC_PREFIX=/path/to/slepc-prefix \
    -DTAILOR_YAML_CPP_PREFIX=/path/to/yaml-cpp-prefix \
    -DTAILOR_TEST_PYTHON=/path/to/python \
    -DTAILOR_MPIEXEC=/path/to/mpiexec
```

CMake 会通过编译检查拒绝 real-scalar PETSc。

## 目录

```text
Tailor/
├── case/
│   ├── config.yaml
│   └── data/
│       ├── sample.h5
│       └── FD-q/fdq_sample_qy10_qz6.h5
├── fdqNodes/                 # 可复用的一维 FD-q 规则缓存
├── src/
│   ├── CPlusPlus/            # PETSc/SLEPc 求解器
│   └── Python/               # 插值、缓存和运行入口
├── tests/
├── CMakeLists.txt
└── Tailor.py
```

仓库自带的 `case/` 是一个小型可运行示例。周期展向采用半开区间唯一节点，不保存
重复末端点。

## Case 配置

`case/config.yaml` 包含输入、离散、物理参数、特征值搜索和输出设置。路径相对于
配置文件所在目录解析：

```yaml
CaseTitle: Sample

Folder: data
File: sample.h5

Q-Value:
    y: 10
    z: 6

Stability:
    Alpha:
        Real: 2.43
        Imag: 0.0

EigenSolver:
    SearchCenterOmega:
        Real: 0.0
        Imag: 0.0
    NumberOfEigenvalues: 8
    Tolerance: 1.0e-10
    MaximumIterations: 1000

Output:
    File: data/results/sample_eigenmodes.h5
```

输运模型固定为 Sutherland 定律，因此无需模型选择字段；仍需在 `Physics` 中提供
Reynolds 数、Mach 数、Prandtl 数、比热比、参考温度、参考黏度和 Sutherland
常数。

## 运行

官方入口始终先验证或自动更新 FD-q 缓存，再启动 C++ 程序。

仅准备输入：

```bash
python Tailor.py -c case/config.yaml --prepare-only
```

开发机只装配矩阵和边界、不运行特征值求解：

```bash
python Tailor.py -c case/config.yaml -n 2 -- -tailor_assemble_only
```

在计算环境中求解局部谱：

```bash
python Tailor.py -c case/config.yaml -n 8 -- \
    -st_pc_factor_mat_solver_type mumps
```

`--` 之后的参数会原样传递给 PETSc/SLEPc。可以用 `--solver`、
`--mpiexec`、`TAILOR_EXECUTABLE` 和 `TAILOR_MPIEXEC` 覆盖自动发现结果。

## 边界条件

- 壁面：保留密度连续性方程，并施加 \(u'=v'=w'=T'=0\)；
- 远场：沿曲线网格外法向执行局部线性特征分解，入射模设为零，出射模和近零模
  保留投影后的完整方程；
- 展向：周期 DMDA、周期 Ghost 和环绕 FD-q stencil。

## 输出

成功求解后，示例 case 写入：

```text
case/data/results/sample_eigenmodes.h5
```

输出目录会自动创建，文件通过临时文件验证后原子替换。HDF5 包含：

```text
/grid
/baseflow
/spectrum/lambda
/spectrum/omega
/spectrum/residual
/modes/mode_000
/modes/mode_001
...
```

网格、基本流和模态使用 DMDA natural ordering：

```text
ordering = k_j_dof
mode dof = [rho, u, v, w, T]
```

## 测试

运行 Python 单元测试：

```bash
python -m unittest discover -s tests/Python -v
```

运行 C++、MPI 和 Python/C++ 集成测试：

```bash
ctest --test-dir build --output-on-failure
```

默认测试 case 是仓库内的 `case/config.yaml`。需要使用其他 case 或 Python
解释器时，可在 CMake 配置阶段设置 `TAILOR_TEST_CONFIG` 和
`TAILOR_TEST_PYTHON`。
