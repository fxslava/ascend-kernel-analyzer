"""Audited arch_3510 logical placements and UB partition contract."""

from dataclasses import dataclass


ASCEND_950PR_POSITIONS = frozenset({"VECIN", "VECOUT", "GM", "TSCM"})


@dataclass(frozen=True)
class UBPartition:
    static_bytes: int
    dynamic_bytes: int

    @property
    def allocated_bytes(self) -> int:
        return self.static_bytes + self.dynamic_bytes

    @property
    def datacache_bytes(self) -> int:
        return min(262144 - self.static_bytes - self.dynamic_bytes - 8192, 131072)

    @property
    def valid(self) -> bool:
        return self.datacache_bytes >= 32768 and self.allocated_bytes <= 221184
