# Tailor

Tailor 是一个基于 complex PETSc/SLEPc 的时域 BiGlobal 稳定性求解器。当前入口将
Python 数据预处理与 C++ PETSc 程序串联为一次完整运行：

```text
Tailor.py
  -> 解析 case/config.yaml
  -> 检查或生成 FD-q PETSc HDF5 缓存
  -> 启动 C++ tailor 求解器
```

目前 C++ 阶段已经能够并行加载网格和基流、建立 DMDA，并读取 FD-q 两个方向的
计算节点、stencil 索引以及零至二阶微分权重。加载阶段会校验 HDF5 schema、数组
形状、索引范围和低阶多项式矩。随后程序计算一阶和二阶曲线坐标 metrics，以及
基本流的 \(y,z,yy,zz,yz\) 导数。物理空间的
\(\Gamma,A,B,C,D,V_{xx},V_{xy},V_{xz},V_{yy},V_{yz},V_{zz}\)
节点系数也已按 `mod_cubes.f90` 实现，并可按需即时生成；坐标系数变换、全局矩阵
组装和特征值求解将在后续阶段继续实现。

## 环境与构建

项目默认使用：

- complex PETSc/SLEPc：`/Users/becrazy/Wro/petsc/arch-complex`
- Python 环境：`/Users/becrazy/Wro/petsc-python`
- HDF5、yaml-cpp、NumPy、SciPy、h5py、petsc4py 和 slepc4py

在项目根目录构建 C++ 程序：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

如安装位置不同，可在配置 CMake 时覆盖 `TAILOR_PETSC_PREFIX` 和
`TAILOR_YAML_CPP_PREFIX`。CMake 会拒绝 real-scalar PETSc 或来自其他 MPI 安装的
编译器，避免混合链接。

## Case 配置

程序要求显式传入 YAML 配置文件。与 FD-q 预处理相关的基本字段为：

```yaml
CaseTitle: Sample

Folder: data
File: sample.h5

Q-Value:
    y: 10
    z: 6
```

输入文件由 `Folder` 和 `File` 相对于配置文件所在目录解析。以上配置对应：

```text
输入：data/sample.h5
输出：data/FD-q/fdq_sample_qy10_qz6.h5
```

输运模型固定采用 Sutherland 定律，因此配置中不需要 `Physics.Transport.Model`；
仍需提供 `ReferenceMu`、`ReferenceTemperature` 和 `SutherlandConstant`。

## 完整运行

先激活 Python 环境：

```bash
source /Users/becrazy/Wro/petsc-python/bin/activate
```

串行运行：

```bash
python Tailor.py -c /Users/becrazy/Wro/cases/global/config.yaml
```

使用两个 MPI 进程运行：

```bash
python Tailor.py -c /Users/becrazy/Wro/cases/global/config.yaml -n 2
```

Python 主程序首先在独立子进程中检查或生成 FD-q 文件。缓存有效时直接复用，随后
自动启动 C++ 求解器。任一阶段失败时，`Tailor.py` 会将非零返回码传回终端。

## PETSc 参数

在 `--` 后添加的参数会原样传递给 C++ PETSc/SLEPc 程序：

```bash
python Tailor.py -c /Users/becrazy/Wro/cases/global/config.yaml -n 2 -- \
    -eps_nev 8 -eps_tol 1e-10
```

## 仅执行预处理

只生成或验证 FD-q 缓存，不启动 C++：

```bash
python Tailor.py -c /Users/becrazy/Wro/cases/global/config.yaml --prepare-only
```

## 指定 C++ 与 MPI 程序

默认依次在 `build/`、`cmake-build-debug/`、多配置构建目录及 `PATH` 中查找
`tailor`。可以显式指定：

```bash
python Tailor.py -c config.yaml --solver /path/to/tailor
```

也可以设置环境变量 `TAILOR_EXECUTABLE`。

当 `-n` 大于 1 时，默认使用 complex PETSc 安装内的 `mpiexec`，不会自动选择系统
中的其他 MPI。需要覆盖时使用：

```bash
python Tailor.py -c config.yaml -n 2 --mpiexec /path/to/mpiexec
```

或设置环境变量 `TAILOR_MPIEXEC`。

## 测试

运行 Python 测试：

```bash
python -m unittest discover -s tests/Python -v
```

运行 CMake/CTest 测试（测试构建需要 `gfortran`，用于从规范 Fortran 公式生成
LNS 系数黄金数据；测试运行本身不调用 Fortran）：

```bash
ctest --test-dir build --output-on-failure
```

## 辅助工具

`tools/` 保存可复现的数据准备和 PETSc HDF5 读取验证程序，默认不参与构建。需要时：

```bash
cmake -S . -B build -DTAILOR_BUILD_TOOLS=ON
cmake --build build -j
```
