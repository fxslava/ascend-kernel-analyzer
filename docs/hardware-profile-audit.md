# Hardware profile audit — evidence and unresolved fields

An accurate audit cannot certify undocumented parameters. The registry accepts
SKU-specific JSON overrides and exposes uncertain descriptions; it is not a
replacement for a CANN platform export or target calibration.

| Target | UB / L1 / L0A / L0B / L0C, KiB | Evidence/status |
|---|---|---|
| 310P AI Core | 256 / 1024 / 64 / 64 / 128 | New provisional profile. These are modeling defaults requiring SKU verification; does not describe 310P Vector Core |
| 910B | 192 / 512 / 64 / 64 / 128 | Existing local `NPUs/Ascend910B4.md` table supports the B4 interpretation; that untracked table has no independently verified provenance. Do not generalize to every B alias |
| 910C | 256 / 512 / 64 / 64 / 256 | Existing extrapolated profile remains provisional. No exact per-SKU primary capacity confirmation obtained |
| 351x | 256 / 512 / 64 / 64 / 128 | Existing provisional profile retained; specific v3 target requires an export |
| 950PR / dav-3510 | 256 / 512 / 64 / 64 / 256 | Existing provisional profile retained. It must not inherit dav-c310 facts merely because filenames are similar |

910B's historical `provisional=False` refers to the local capacity table; it
does not certify bandwidth, banking, all B variants, or queue depth. Changing
that compatibility flag alone would not solve field-specific provenance. The
report explicitly distinguishes its limits.

## Primary evidence reviewed

1. Huawei [GetCoreMemSize](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1034.html)
   returns hardware storage capacities in bytes. Export UB, L1, L0A/B/C, BT and
   FB for the selected SKU and use `domains` overrides. This supports the query
   procedure, not the table's numerical defaults.
2. Huawei [SetFlag/WaitFlag](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html)
   describes same-core event synchronization, paired set/wait usage, static
   event restrictions and enumerated route names. The API lists A2/A3 and
   inference AI Core event ranges 0..7 and reserves 6/7 in static programming.
   This does **not** make every enumerated route legal on every SKU or bridge
   AIC and AIV event spaces.
3. Huawei [static tensor programming](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)
   demonstrates distinct event IDs and prologue priming for ping-pong buffers.
4. Huawei [L1-to-GM DataCopy](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/DataCopy_L1ToGM_continuous.md)
   documents 32-byte L1 source/count alignment for that API, with separate GM
   requirements and target applicability. Alignment must be API-specific;
   a 32-byte block is not a universal tensor-format alignment requirement.

## Routes, queues, topology and banks

`supported_routes=None` means applicability is unknown. An explicit route
allowlist enables fatal AKA2011 rejection of unsupported routes. The generic
parser remains syntax-only; parsing `FIX_MTE3` cannot establish availability.
Complete validated per-target route lists were not available in this audit.

`queue_depths` is an explicit positive-integer metadata map. Empty means unknown.
Event IDs, TQue buffer count, prefetch tracking slots, SRAM capacity and hardware
instruction queue depth are different quantities. No depth is inferred from
one of the others. Flow buffer capacities are separate calibrated byte-equivalent
values and are never synthesized by multiplying an invented slot depth.

Topology is `unified` for the provisional 310P AI Core profile and `split` for
the existing A2/A3 descriptions. Exact core counts and per-instance bandwidth
are SKU-specific and not inferred by the solver. Local interval resource keys
must include the actual core instance; GM resource keys must identify shared
storage. Local HardEvents are not proof of intercore GM ordering.

Bank checks retain the existing logical-group model: block-index modulo
`ub_bank_count`, with an explicit port-count parameter. Two read ports suppress
the existing dual-source contention warning. The B4 local description mentions
16 groups/64 physical banks; those are not interchangeable with verified UB
capacity. Swizzles, per-lane bank mapping, broadcasts and port arbitration are
not fully characterized. Warnings now describe potential contention under the
logical model instead of asserting a measured stall.

Rates, queue capacities, bank ports, route applicability, and Cube granules are
profile inputs; nonpositive/nonfinite flow rates/capacities and invalid basic
geometry are rejected. Cube cost now uses the profile's granule and MAC throughput.
Default analytical byte rates remain historical estimates. The extra 30-cycle
flag delay was removed because it had no calibration evidence.

Logical position and transfer-path defaults now reside in the processor
description. JSON `position_domains`, `transfer_pipes` and `pipes` overrides
control the hardware facade, AST transfer selection, legality checking and
global barrier participants. Huawei API versions can map logical positions or
supported paths differently across targets. Full target/version-dependent
intrinsic validation is a remaining frontend metadata task. Do not interpret this
audit as confirmation of all existing ISA/path assumptions.

## Deployment acceptance criteria

Obtain a versioned target export, record CANN version and exact SKU/core type,
override domain capacities/alignments, supply validated route allowlists and
queue depths, calibrate rates/delays, and verify bank topology against target
documentation or microbenchmarks. Until then 310P/910C/351x remain provisional,
and 910B claims are limited to the available B4 local description.
