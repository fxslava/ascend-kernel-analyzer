"""Synthetic tests for the numerical-parity decision procedures.

Each rule is driven here as pure data - small marked graphs, split windows,
store probes - with no C++ or MLIR frontend involved, so a regression in the
decision logic is visible independently of parsing.
"""

import networkx as nx

from ascend_analyzer.backend.parity import (
    GmStoreProbe,
    LoopProbe,
    MergeTreeTracker,
    SplitWindow,
    SyncPairProbe,
    invariant_flush_loops,
    replay_halving_chain,
    vacuous_sync_pairs,
)


def pair(set_index, wait_index, src="PIPE_MTE2", dst="PIPE_V"):
    return SyncPairProbe(
        set_index=set_index,
        wait_index=wait_index,
        src_pipe=src,
        dst_pipe=dst,
        route=f"{src[5:]}_{dst[5:]}",
        event_id=0,
    )


def window(half_count, tensor="buf", index=0, line=1, kind="gather_pair"):
    return SplitWindow(
        tensor=tensor, half_count=half_count, index=index, line=line, kind=kind
    )


class TestVacuousSyncPairs:
    def test_canonical_drain_idiom_is_meaningful(self):
        # MTE2: copy(1) -> set(2);   V: wait(3); the only path 1 -> 3 is the
        # pair's own edge, so the handshake orders the copy. No verdict.
        graph = nx.DiGraph([(1, 2), (2, 3)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(2, 3)],
            pipe_ops={"PIPE_MTE2": [1, 2], "PIPE_V": [3]},
            work={"PIPE_MTE2": [1], "PIPE_V": [3]},
        )
        assert verdicts == []

    def test_set_with_no_producer_work_is_vacuous(self):
        graph = nx.DiGraph([(2, 3)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(2, 3)],
            pipe_ops={"PIPE_MTE2": [2], "PIPE_V": [3]},
            work={"PIPE_MTE2": [], "PIPE_V": []},
        )
        assert len(verdicts) == 1
        assert verdicts[0].kind == "no_producer_work"

    def test_work_already_ordered_elsewhere_is_nothing_new(self):
        # The copy reaches the wait through an earlier handshake
        # (1 -> 4 -> 3), so removing the pair's own edge 2 -> 3 leaves the
        # work ordered: the pair adds nothing.
        graph = nx.DiGraph([(1, 2), (1, 4), (4, 3), (2, 3)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(2, 3)],
            pipe_ops={"PIPE_MTE2": [1, 2, 4], "PIPE_V": [3]},
            work={"PIPE_MTE2": [1]},
        )
        assert len(verdicts) == 1
        assert verdicts[0].kind == "nothing_new"
        assert verdicts[0].outstanding_before == 1

    def test_producer_hint_names_the_pipe_with_the_work(self):
        graph = nx.DiGraph([(2, 3), (1, 10)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(2, 3)],
            pipe_ops={"PIPE_MTE2": [2], "PIPE_V": [1, 3]},
            work={"PIPE_MTE2": [], "PIPE_V": [1]},
        )
        assert verdicts[0].producer_hint == ("PIPE_V",)

    def test_an_op_on_the_waiting_pipe_inside_the_pair_breaks_adjacency(self):
        graph = nx.DiGraph([(1, 2), (2, 4), (3, 4)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(2, 4)],
            pipe_ops={"PIPE_MTE2": [1, 2], "PIPE_V": [3, 4]},
            work={"PIPE_MTE2": [1]},
        )
        assert verdicts == []

    def test_loop_carried_pair_is_not_examined(self):
        graph = nx.DiGraph([(1, 2)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(4, 2)],  # wait precedes its set: the previous iteration's
            pipe_ops={"PIPE_MTE2": [4], "PIPE_V": [2]},
            work={},
        )
        assert verdicts == []

    def test_prologue_prime_pair_is_a_ring_closure(self):
        # set(1) sits before the loop whose body opens with wait(4).
        graph = nx.DiGraph([(1, 4)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(1, 4)],
            pipe_ops={"PIPE_MTE2": [1], "PIPE_V": [4]},
            work={"PIPE_V": []},
            loop_spans=[(4, 20)],
        )
        assert verdicts == []

    def test_epilogue_drain_pair_is_a_ring_closure(self):
        graph = nx.DiGraph([(10, 30)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(10, 30)],
            pipe_ops={"PIPE_MTE2": [10], "PIPE_V": [30]},
            work={"PIPE_MTE2": []},
            loop_spans=[(4, 20)],
        )
        assert verdicts == []

    def test_intra_body_pair_is_still_examined_despite_loops(self):
        graph = nx.DiGraph([(6, 7)])
        verdicts = vacuous_sync_pairs(
            graph,
            [pair(6, 7)],
            pipe_ops={"PIPE_MTE2": [6], "PIPE_V": [7]},
            work={"PIPE_MTE2": []},
            loop_spans=[(4, 20)],
        )
        assert len(verdicts) == 1
        assert verdicts[0].kind == "no_producer_work"


class TestMergeTreeTracker:
    def test_full_descent_to_the_floor_is_clean(self):
        tracker = MergeTreeTracker(8)
        assert tracker.check_window(window(128)) is None  # [8, 32]
        tracker.apply_halving("buf", 128)
        assert tracker.check_window(window(64)) is None  # [8, 16]
        tracker.apply_halving("buf", 64)
        assert tracker.check_window(window(32)) is None  # [8, 8]
        tracker.apply_halving("buf", 32)
        assert tracker.check_window(window(16)) is None  # [8, 4]
        tracker.apply_halving("buf", 16)
        assert tracker.check_window(window(8)) is None  # [8, 2] - the floor
        tracker.apply_halving("buf", 8)
        violation = tracker.check_window(window(4))  # [8, 1]: rows pair up
        assert violation is not None
        assert violation.kind == "cross_row"
        assert violation.length == 8

    def test_ragged_window_is_flagged(self):
        tracker = MergeTreeTracker(8)
        violation = tracker.check_window(window(10))  # 20 elements, not 8k
        assert violation is not None
        assert violation.kind == "ragged"

    def test_window_below_the_floor_without_dividing_is_cross_row(self):
        tracker = MergeTreeTracker(8)
        violation = tracker.check_window(window(3))  # 6 elements < 16
        assert violation.kind == "cross_row"

    def test_no_row_anchor_means_no_verdict(self):
        tracker = MergeTreeTracker(None)
        assert tracker.check_window(window(1)) is None

    def test_single_row_layout_is_not_a_batch(self):
        tracker = MergeTreeTracker(1)
        assert tracker.check_window(window(2)) is None


class TestReplayHalvingChain:
    def test_fixed_kernel_bound_stops_at_two_rows(self):
        body, tail = replay_halving_chain(256, 16)
        assert body == [256, 128, 64, 32]
        assert tail == 16

    def test_over_descended_bound_ends_below_the_floor(self):
        body, tail = replay_halving_chain(256, 8)
        assert body == [256, 128, 64, 32, 16]
        assert tail == 8  # < 2 * rows for an 8-row layout

    def test_non_halving_divisor(self):
        body, tail = replay_halving_chain(81, 3, divisor=3)
        assert body == [81, 27, 9]
        assert tail == 3

    def test_inclusive_bound(self):
        body, tail = replay_halving_chain(64, 16, strict=False)
        assert body == [64, 32, 16]
        assert tail == 8


class TestInvariantFlushLoops:
    def test_invariant_inner_destination_is_flagged(self):
        loops = {
            0: LoopProbe(id=0, var="row0", parent=None, trip_count=None, line=10),
            1: LoopProbe(id=1, var="col0", parent=0, trip_count=None, line=12, accumulates=True),
        }
        store = GmStoreProbe(
            index=5, line=30, dst_text="legOutGm_[legIndex][row0]",
            store_name="DataCopyPad", loop_id=1,
        )
        verdicts = invariant_flush_loops([store], loops)
        assert len(verdicts) == 1
        assert verdicts[0].reduction is True
        assert verdicts[0].mentioned == ("row0",)

    def test_destination_advancing_with_its_loop_is_clean(self):
        loops = {1: LoopProbe(id=1, var="col0", parent=0, trip_count=None, line=12)}
        store = GmStoreProbe(
            index=5, line=30, dst_text="out[col0]", store_name="DataCopy", loop_id=1,
        )
        assert invariant_flush_loops([store], loops) == []

    def test_single_trip_loop_is_clean(self):
        loops = {1: LoopProbe(id=1, var="c", parent=None, trip_count=1, line=12)}
        store = GmStoreProbe(
            index=5, line=30, dst_text="out", store_name="DataCopy", loop_id=1,
        )
        assert invariant_flush_loops([store], loops) == []

    def test_atomic_accumulation_is_clean(self):
        loops = {1: LoopProbe(id=1, var="c", parent=None, trip_count=None, line=12)}
        store = GmStoreProbe(
            index=5, line=30, dst_text="out", store_name="DataCopy", loop_id=1,
            atomic=True,
        )
        assert invariant_flush_loops([store], loops) == []

    def test_kernel_level_store_is_clean(self):
        loops = {1: LoopProbe(id=1, var="c", parent=None, trip_count=None, line=12)}
        store = GmStoreProbe(
            index=5, line=30, dst_text="out", store_name="DataCopy", loop_id=None,
        )
        assert invariant_flush_loops([store], loops) == []

    def test_conditional_store_is_clean(self):
        loops = {1: LoopProbe(id=1, var="c", parent=None, trip_count=None, line=12)}
        store = GmStoreProbe(
            index=5, line=30, dst_text="out", store_name="DataCopy", loop_id=1,
            conditional=True,
        )
        assert invariant_flush_loops([store], loops) == []

    def test_plain_loop_without_reduction_still_reports(self):
        loops = {1: LoopProbe(id=1, var="c", parent=None, trip_count=None, line=12)}
        store = GmStoreProbe(
            index=5, line=30, dst_text="out", store_name="DataCopy", loop_id=1,
        )
        verdicts = invariant_flush_loops([store], loops)
        assert len(verdicts) == 1
        assert verdicts[0].reduction is False

    def test_induction_var_matched_on_word_boundaries(self):
        loops = {1: LoopProbe(id=1, var="col", parent=None, trip_count=None, line=12)}
        # "cols" is a different identifier: it does not advance with "col",
        # so the store is invariant and must be reported.
        lookalike = GmStoreProbe(
            index=5, line=30, dst_text="out[cols + 1]", store_name="DataCopy", loop_id=1,
        )
        assert len(invariant_flush_loops([lookalike], loops)) == 1
        exact = GmStoreProbe(
            index=5, line=30, dst_text="out[col]", store_name="DataCopy", loop_id=1,
        )
        assert invariant_flush_loops([exact], loops) == []
