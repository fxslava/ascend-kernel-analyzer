---
name: ascend-kernel-verification-protocol
description: Verify Ascend 950PR arch35 AIV kernels with PipeSim tile budgets, ascend_analyzer, and BiSheng device object compilation while avoiding host kernel execution and full tensor cycle simulation.
---

# Ascend 950PR kernel verification protocol

Apply this gate to `ascend-moe-engine` KER-01 through KER-05 and comparable `dav-3510` AIV kernels. The gate establishes static/layout and compilation validity only; report those limits accurately.

1. Before editing a tile, inspect the Ascend 950PR PipeSim profile with `npu-pipesim archs ascend950pr` and model the proposed tile. Keep static UB at or below `216 * 1024` bytes per AIV: 256 KiB physical less at least 32 KiB SIMT DataCache and 8 KiB compiler reserve. Record the byte budget and add a matching `static_assert`.
2. Audit each kernel with `ascend_analyzer <kernel.cpp> --chip ascend950pr --ub-bytes 221184 --warnings-as-errors`. Resolve every fatal or warning and regenerate any layout annotations before repeating the audit.
3. Compile each device translation unit inside `cann92_dev` as the workspace UID/GID after sourcing `/usr/local/Ascend/ascend-toolkit/set_env.sh`: `bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang -c <kernel.cpp> -o <kernel.dav-3510.o>`. Supply local include paths as needed. Check exit code and object existence. Use the actual mounted work directory; in this environment the repository is mounted at `/home/ascend-moe-engine` rather than `/engine`.
4. For hybrid kernels, confirm `SPLIT_CORE_VEC` and `__DAV_VEC__` remain defined so `asc_vf_call` does not disappear. Preserve `__simt_vf__` and vector-core split semantics.

Never run host CPU test harnesses, `ctest`, `cpudebug`, `tikicpulib`, `libascendc_cpudebug.so`, CPU emulators, or full-scale `CAModel` tensor cycle simulation for this gate. Do not run or simulate kernel execution. Stop verification after the analyzer and clean device compilation; runtime numerical correctness and device performance remain unverified.

For arch35, only `EVENT_ID0` through `EVENT_ID3` are legal. Pair targeted flags and include `MTE3_MTE2` where a state reload follows a state write. Use SIMD or register SIMT for bulk tensor work, never scalar `GetValue`/`SetValue` loops over tensor buffers. For MXFP4, decode low nibble first and read packed bytes from GM through SIMT DataCache without transient UB LUTs.