/* Tile geometry for the cube_tbuf_pipeline fixture.
 *
 * Deliberately supplied as a local header next to the kernel: including it
 * exercises the #include "..." inlining of the macro expander, so the layout
 * folder sees the full #define chain (TILE_A_BYTES -> M * K / 2). */
#ifndef CUBE_TBUF_GEOMETRY_H
#define CUBE_TBUF_GEOMETRY_H

#define M            16
#define N            16
#define K            64
#define TILES        4

#define TILE_ELEMS   (M * N)
#define TILE_A_BYTES (M * K / 2)   /* 16x64 packed FP4 = 512 B */
#define TILE_B_BYTES (K * N / 2)   /* 64x16 packed FP4 = 512 B */
#define TILE_C_BYTES (TILE_ELEMS * 4)

#endif  /* CUBE_TBUF_GEOMETRY_H */
