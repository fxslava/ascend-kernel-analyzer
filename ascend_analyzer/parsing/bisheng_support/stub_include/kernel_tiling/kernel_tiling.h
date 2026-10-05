// Declaration-only facade for CANN tiling; never used to generate device code.
#pragma once
#include <cstdint>
namespace AscendC {
struct TCubeTiling {
    uint32_t usedCoreNum, M, N, Ka, Kb, singleCoreM, singleCoreN, singleCoreK;
    uint32_t baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb;
};
struct TilingData { TCubeTiling cubeTiling; };
struct ShapeInfo { uint32_t m, n, k; };
}
using AscendC::TCubeTiling;
#define GET_TILING_DATA_WITH_STRUCT(Type, name, ptr) Type name = *reinterpret_cast<const Type *>(ptr)
#define REGISTER_TILING_DEFAULT(Type)
#define GET_TILING_DATA(name, ptr) AscendC::TilingData name = *reinterpret_cast<const AscendC::TilingData *>(ptr)
