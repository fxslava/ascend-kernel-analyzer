"""Tests for the hardware model: chip profiles, pipes, routes and domains."""

from __future__ import annotations

import json

import pytest

from ascend_analyzer.hardware import (
    CHIP_PROFILES,
    KIB,
    HardEventRoute,
    HardwareModel,
    PhysicalDomain,
    Pipe,
    TPosition,
    resolve_chip,
)


class TestChipResolution:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("ascend910b", "ascend910b"),
            ("910b", "ascend910b"),
            ("Atlas-A2", "ascend910b"),
            ("A2", "ascend910b"),
            ("ascend910c", "ascend910c"),
            ("910C", "ascend910c"),
            ("351x", "ascend351x"),
            ("ascend_351x", "ascend351x"),
        ],
    )
    def test_resolves_aliases_case_insensitively(self, name, expected):
        assert resolve_chip(name).name == expected

    def test_unknown_chip_lists_the_known_ones(self):
        with pytest.raises(KeyError) as excinfo:
            resolve_chip("ascend999z")
        message = str(excinfo.value)
        assert "ascend910b" in message

    def test_910b_ub_is_192_kib(self):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.capacity(PhysicalDomain.UB) == 192 * KIB

    def test_910c_ub_is_larger_than_910b(self):
        b = HardwareModel.for_chip("ascend910b")
        c = HardwareModel.for_chip("ascend910c")
        assert c.capacity(PhysicalDomain.UB) > b.capacity(PhysicalDomain.UB)

    def test_provisional_profiles_are_flagged(self):
        assert not CHIP_PROFILES["ascend910b"].provisional
        assert CHIP_PROFILES["ascend910c"].provisional
        assert CHIP_PROFILES["ascend351x"].provisional
        # A provisional profile must say why, so the report can pass that on.
        assert CHIP_PROFILES["ascend910c"].notes


class TestAlignment:
    def test_ub_and_l1_align_to_one_32_byte_block(self):
        hw = HardwareModel.for_chip("ascend910b")
        for domain in (PhysicalDomain.UB, PhysicalDomain.L1):
            assert hw.base_alignment(domain) == 32
            assert hw.size_alignment(domain) == 32
            assert hw.stride_alignment(domain) == 32

    def test_cube_buffers_align_to_a_fractal(self):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.base_alignment(PhysicalDomain.L0A) == 512
        assert hw.base_alignment(PhysicalDomain.L0B) == 512
        assert hw.base_alignment(PhysicalDomain.L0C) == 1024

    def test_unknown_domain_falls_back_to_the_block_size(self):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.base_alignment(PhysicalDomain.UNKNOWN) == hw.chip.block_bytes


class TestCapacityOverrides:
    def test_override_replaces_capacity_and_marks_provisional(self):
        hw = HardwareModel.for_chip("ascend910b")
        patched = hw.with_capacity_overrides({PhysicalDomain.UB: 64 * KIB})
        assert patched.capacity(PhysicalDomain.UB) == 64 * KIB
        assert patched.chip.provisional
        # Other domains and the original model are untouched.
        assert patched.capacity(PhysicalDomain.L1) == 512 * KIB
        assert hw.capacity(PhysicalDomain.UB) == 192 * KIB

    def test_empty_override_returns_the_same_model(self):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.with_capacity_overrides({}) is hw
        assert hw.with_capacity_overrides({PhysicalDomain.UB: None}) is hw

    def test_profile_file_round_trip(self, tmp_path):
        profile = tmp_path / "custom.json"
        profile.write_text(
            json.dumps(
                {
                    "name": "my910b",
                    "base": "ascend910b",
                    "reserved_event_ids": [7],
                    "max_event_id": 7,
                    "domains": {"UB": {"capacity_bytes": 100000, "base_alignment": 64}},
                }
            ),
            encoding="utf-8",
        )
        hw = HardwareModel.from_profile_file(profile)
        assert hw.chip.name == "my910b"
        assert hw.capacity(PhysicalDomain.UB) == 100000
        assert hw.base_alignment(PhysicalDomain.UB) == 64
        # Unspecified domains are inherited from the base profile.
        assert hw.capacity(PhysicalDomain.L1) == 512 * KIB
        assert not hw.is_event_id_reserved(6)
        assert hw.is_event_id_reserved(7)


class TestEventIds:
    def test_ids_6_and_7_are_reserved_by_default(self):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.is_event_id_reserved(6)
        assert hw.is_event_id_reserved(7)
        for usable in range(6):
            assert not hw.is_event_id_reserved(usable)

    def test_range_check(self):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.is_event_id_in_range(0)
        assert hw.is_event_id_in_range(7)
        assert not hw.is_event_id_in_range(8)
        assert not hw.is_event_id_in_range(-1)


class TestPipeParsing:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("PIPE_MTE2", Pipe.MTE2),
            ("MTE2", Pipe.MTE2),
            ("pipe_t::PIPE_V", Pipe.V),
            ("AscendC::PIPE_ALL", Pipe.ALL),
            ("mte3", Pipe.MTE3),
            ("FIX", Pipe.FIX),
        ],
    )
    def test_parses_every_spelling(self, text, expected):
        assert Pipe.parse(text) is expected

    @pytest.mark.parametrize("text", ["", "PIPE_NOPE", "VECTOR", "   "])
    def test_rejects_nonsense(self, text):
        assert Pipe.parse(text) is None

    def test_pipe_all_is_not_a_real_pipe(self):
        assert not Pipe.ALL.is_real
        assert Pipe.V.is_real


class TestHardEventRoutes:
    @pytest.mark.parametrize(
        "text, src, dst",
        [
            ("MTE2_V", Pipe.MTE2, Pipe.V),
            ("V_MTE3", Pipe.V, Pipe.MTE3),
            ("MTE3_MTE2", Pipe.MTE3, Pipe.MTE2),
            ("AscendC::HardEvent::MTE2_V", Pipe.MTE2, Pipe.V),
            ("HardEvent::M_MTE1", Pipe.M, Pipe.MTE1),
            ("S_V", Pipe.S, Pipe.V),
            ("FIX_V", Pipe.FIX, Pipe.V),
        ],
    )
    def test_route_direction_is_src_to_dst(self, text, src, dst):
        route = HardEventRoute.parse(text)
        assert route is not None
        assert route.src is src
        assert route.dst is dst

    @pytest.mark.parametrize("text", ["MTE2", "", "PIPE_ALL_V", "NOPE_NOPE"])
    def test_rejects_malformed_routes(self, text):
        assert HardEventRoute.parse(text) is None

    def test_self_route_is_detected(self):
        assert HardEventRoute.from_pipes(Pipe.V, Pipe.V).is_self_route
        assert not HardEventRoute.from_pipes(Pipe.V, Pipe.MTE3).is_self_route

    def test_from_pipes_builds_the_canonical_name(self):
        assert HardEventRoute.from_pipes(Pipe.MTE2, Pipe.V).name == "MTE2_V"
        assert HardEventRoute.from_pipes(Pipe.MTE3, Pipe.MTE2).name == "MTE3_MTE2"

    def test_parse_is_inverse_of_from_pipes(self):
        route = HardEventRoute.from_pipes(Pipe.MTE3, Pipe.MTE1)
        assert HardEventRoute.parse(route.name) == route


class TestTPositionMapping:
    @pytest.mark.parametrize(
        "position, domain",
        [
            (TPosition.VECIN, PhysicalDomain.UB),
            (TPosition.VECOUT, PhysicalDomain.UB),
            (TPosition.VECCALC, PhysicalDomain.UB),
            (TPosition.CO2, PhysicalDomain.UB),
            (TPosition.A1, PhysicalDomain.L1),
            (TPosition.B1, PhysicalDomain.L1),
            (TPosition.TSCM, PhysicalDomain.L1),
            (TPosition.A2, PhysicalDomain.L0A),
            (TPosition.B2, PhysicalDomain.L0B),
            (TPosition.CO1, PhysicalDomain.L0C),
            (TPosition.GM, PhysicalDomain.GM),
        ],
    )
    def test_logical_position_maps_to_physical_domain(self, position, domain):
        hw = HardwareModel.for_chip("ascend910b")
        assert hw.domain_of(position) is domain

    def test_parses_qualified_names(self):
        assert TPosition.parse("AscendC::TPosition::VECIN") is TPosition.VECIN
        assert TPosition.parse("vecout") is TPosition.VECOUT
        assert TPosition.parse("NOT_A_POSITION") is None

    def test_only_on_core_buffers_are_capacity_constrained(self):
        assert PhysicalDomain.UB.is_on_core_sram
        assert PhysicalDomain.L0C.is_on_core_sram
        assert not PhysicalDomain.GM.is_on_core_sram
        assert not PhysicalDomain.UNKNOWN.is_on_core_sram


def test_describe_is_json_serialisable():
    hw = HardwareModel.for_chip("ascend910b")
    payload = hw.describe()
    json.dumps(payload)  # must not raise
    assert payload["domains"]["UB"]["capacity_bytes"] == 192 * KIB
    assert payload["reserved_event_ids"] == [6, 7]


@pytest.mark.parametrize("alias", ["ascend950pr", "dav-3510", "Ascend950", "Ascend910_9589"])
def test_950pr_target_invariants(alias):
    hw = HardwareModel.for_chip(alias)
    assert hw.capacity(PhysicalDomain.UB) == 256 * KIB
    assert hw.chip.reserved_event_ids == frozenset({0, 1, 2})
    assert hw.chip.l0c_base_offsets == (0, 1024)
    assert hw.chip.features == frozenset({"mx_cube", "simt"})
    assert not hw.is_event_id_reserved(3)
    assert not hw.is_event_id_reserved(4)


def test_c220_target_invariants():
    hw = HardwareModel.for_chip("dav-c220")
    assert hw.capacity(PhysicalDomain.UB) == 192 * KIB
    assert hw.capacity(PhysicalDomain.L1) == 512 * KIB
    assert hw.capacity(PhysicalDomain.L0A) == 64 * KIB
    assert hw.capacity(PhysicalDomain.L0B) == 64 * KIB
    assert hw.capacity(PhysicalDomain.L0C) == 128 * KIB
    assert hw.chip.ub_bank_count == 16
    assert hw.chip.reserved_event_ids == frozenset({6, 7})
    assert not hw.chip.features
