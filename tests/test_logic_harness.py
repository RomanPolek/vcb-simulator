import unittest
from pathlib import Path

from compiler.compiler import VMEM_ADDRESS_KIND, VMEM_DATA_KIND, VMemConfig, VMemLayout
from compiler.extractor import load_project
from main import BLOCKS, build_vmem_config, load_grid_from_project_data
from settings import VCB_SIMULATOR_ROOT
from tests.logic_test_harness import CircuitTestHarness


def two_input_gate(kind_token: str) -> CircuitTestHarness:
    return CircuitTestHarness.from_ascii(
        ". . . . .",
        "- r {} r -".format(kind_token),
        ". . w . .",
        ". . - . .",
    )


def three_input_gate(kind_token: str) -> CircuitTestHarness:
    return CircuitTestHarness.from_ascii(
        ". . - . .",
        ". . r . .",
        "- r {} r -".format(kind_token),
        ". . w . .",
        ". . - . .",
    )


def simple_vmem_harness(initial_words: dict[int, int] | None = None) -> CircuitTestHarness:
    vmem_config = VMemConfig(
        enabled=True,
        address_layout=VMemLayout(bits=1, position_x=2, position_y=1, offset_x=0, offset_y=0, size_x=1, size_y=1),
        data_layout=VMemLayout(bits=1, position_x=2, position_y=3, offset_x=0, offset_y=0, size_x=1, size_y=1),
        persistent_start=1,
        persistent_end=0,
    )
    harness = CircuitTestHarness.from_ascii(
        ". . . . .",
        "- r . w -",
        ". . . . .",
        "- r . w -",
        ". . . . .",
        vmem_config=vmem_config,
        vmem_words=initial_words or {},
    )
    return harness


def vmem_data_latch_harness(initial_words: dict[int, int] | None = None) -> CircuitTestHarness:
    vmem_config = VMemConfig(
        enabled=True,
        address_layout=VMemLayout(bits=1, position_x=2, position_y=1, offset_x=0, offset_y=0, size_x=1, size_y=1),
        data_layout=VMemLayout(bits=1, position_x=2, position_y=3, offset_x=0, offset_y=0, size_x=1, size_y=1),
    )
    harness = CircuitTestHarness.from_ascii(
        ". . . . . . . . .",
        "- r . w - . . . .",
        ". . . . . . . . .",
        ". . . w - r l w -",
        ". . . . . . . . .",
        vmem_config=vmem_config,
        vmem_words=initial_words or {},
    )
    return harness


def multi_read_vmem_latch_harness(target_kind: str, input_count: int) -> tuple[CircuitTestHarness, tuple[tuple[int, int], ...]]:
    if input_count == 2:
        rows = (
            ". . . . .",
            "- r . r -",
            ". . . . .",
            ". . . . .",
            ". . . . .",
        )
        target_position = (2, 1)
        spare_position = (2, 3)
        input_points = ((0, 1), (4, 1))
    elif input_count == 3:
        rows = (
            ". . - . .",
            ". . r . .",
            "- r . r -",
            ". . . . .",
            ". . . . .",
        )
        target_position = (2, 2)
        spare_position = (2, 4)
        input_points = ((2, 0), (0, 2), (4, 2))
    else:
        raise ValueError("input_count must be 2 or 3")

    address_position = target_position if target_kind == "address" else spare_position
    data_position = target_position if target_kind == "data" else spare_position
    vmem_config = VMemConfig(
        enabled=True,
        address_layout=VMemLayout(
            bits=1,
            position_x=address_position[0],
            position_y=address_position[1],
            offset_x=0,
            offset_y=0,
            size_x=1,
            size_y=1,
        ),
        data_layout=VMemLayout(
            bits=1,
            position_x=data_position[0],
            position_y=data_position[1],
            offset_x=0,
            offset_y=0,
            size_x=1,
            size_y=1,
        ),
    )
    return CircuitTestHarness.from_ascii(*rows, vmem_config=vmem_config), input_points


class CircuitTestHarnessTests(unittest.TestCase):
    def test_two_input_truth_tables_for_basic_gates(self):
        cases = {
            "a": {
                (0, 0): 0,
                (0, 1): 0,
                (1, 0): 0,
                (1, 1): 1,
            },
            "o": {
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 1,
                (1, 1): 1,
            },
            "b": {
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 1,
                (1, 1): 1,
            },
            "x": {
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 1,
                (1, 1): 0,
            },
            "q": {
                (0, 0): 1,
                (0, 1): 0,
                (1, 0): 0,
                (1, 1): 0,
            },
            "n": {
                (0, 0): 1,
                (0, 1): 0,
                (1, 0): 0,
                (1, 1): 0,
            },
        }
        input_points = ((0, 1), (4, 1))
        output_point = (2, 3)
        for token, expected in cases.items():
            with self.subTest(token=token):
                harness = two_input_gate(token)
                self.assertEqual(harness.truth_table(input_points, output_point), expected)

    def test_three_input_gate_tracks_distinct_inputs(self):
        harness = three_input_gate("o")
        gate = harness.gate_at((2, 2))
        self.assertEqual(gate.kind, "OR")
        self.assertEqual(len(gate.inputs), 3)

        harness.drive({(2, 0): 1, (0, 2): 0, (4, 2): 0}).tick()
        self.assertEqual(harness.read_net((2, 4)), 1)

    def test_breakpoint_triggers_once_until_input_falls(self):
        harness = two_input_gate("p")

        self.assertFalse(harness.breakpoint_active())

        harness.drive({(0, 1): 1, (4, 1): 0}).tick()
        self.assertTrue(harness.breakpoint_active())
        self.assertFalse(harness.breakpoint_active())

        harness.tick(2)
        self.assertFalse(harness.breakpoint_active())

        harness.drive({(0, 1): 0, (4, 1): 0}).tick()
        self.assertFalse(harness.breakpoint_active())

        harness.drive({(0, 1): 1, (4, 1): 0}).tick()
        self.assertTrue(harness.breakpoint_active())

    def test_multi_block_gate_merges_and_behaves_as_one_gate(self):
        harness = CircuitTestHarness.from_ascii(
            ". . - . . .",
            ". . r . . .",
            "- r a a w -",
            ". . a a . .",
            ". . . . . .",
        )
        gate = harness.gate_at((2, 2))
        self.assertEqual(gate.id, harness.gate_at((3, 3)).id)
        self.assertEqual(gate.kind, "AND")
        self.assertEqual(len(gate.blocks), 4)

        harness.drive({(2, 0): 1, (0, 2): 1}).tick()
        self.assertEqual(harness.read_net((5, 2)), 1)

    def test_same_gate_kind_merges_straight_through_crossroad(self):
        harness = CircuitTestHarness.from_ascii("- w a + a r -")
        gate = harness.gate_at((2, 0))

        self.assertEqual(gate.id, harness.gate_at((4, 0)).id)
        self.assertEqual(gate.kind, "AND")
        self.assertEqual(len(gate.blocks), 2)
        self.assertEqual(len(harness.gates), 1)

        harness.drive({(6, 0): 1}).tick()
        self.assertEqual(harness.read_net((0, 0)), 1)

    def test_zero_input_and_stays_off_while_zero_input_not_turns_on(self):
        harness = CircuitTestHarness.from_ascii(
            "a w - . n w -",
        )
        harness.tick()
        self.assertEqual(harness.read_gate((0, 0)), 0)
        self.assertEqual(harness.read_net((2, 0)), 0)
        self.assertEqual(harness.read_gate((4, 0)), 1)
        self.assertEqual(harness.read_net((6, 0)), 1)

    def test_crossroad_gate_fixture_stays_off_when_one_and_input_is_low(self):
        harness = CircuitTestHarness.from_ascii(
            "- w a + a r a a r w",
            ". . . w r + . w . n",
            ". . . . . . . - . .",
        )
        left_and = harness.gate_at((2, 0))
        downstream_and = harness.gate_at((6, 0))
        self.assertEqual(left_and.kind, "AND")
        self.assertEqual(downstream_and.kind, "AND")
        self.assertEqual(left_and.id, harness.gate_at((4, 0)).id)
        self.assertEqual(len(left_and.inputs), 2)
        self.assertEqual(len(downstream_and.inputs), 2)

        harness.tick(2)

        self.assertEqual(harness.read_gate((9, 1)), 1)
        self.assertEqual(harness.read_net((9, 0)), 1)
        self.assertEqual(harness.read_gate((2, 0)), 0)
        self.assertEqual(harness.read_net((0, 0)), 0)
        self.assertEqual(harness.read_gate((6, 0)), 0)
        self.assertEqual(harness.read_net((7, 2)), 0)

    def test_latch_off_matches_expected_toggle_timeline(self):
        harness = CircuitTestHarness.from_ascii("- r l w -")
        output_timeline = harness.trace_values(
            [
                {(0, 0): 1},
                {(0, 0): 0},
                {(0, 0): 1},
                {(0, 0): 0},
                {(0, 0): 1},
            ],
            (4, 0),
            include_initial=True,
        )
        self.assertEqual(output_timeline, [0, 1, 1, 0, 0, 1])

    def test_latch_detects_rising_edges_per_input_across_ticks(self):
        harness = CircuitTestHarness.from_ascii(
            ". . - . .",
            ". . r . .",
            "- r l w -",
            ". . . . .",
        )
        trace = harness.trace(
            [
                {(0, 2): 1, (2, 0): 0},
                {(0, 2): 0, (2, 0): 0},
                {(0, 2): 0, (2, 0): 1},
                {(0, 2): 1, (2, 0): 0},
            ],
            [(4, 2)],
        )
        self.assertEqual([step[(4, 2)] for step in trace], [1, 1, 0, 1])

    def test_clock_toggles_once_per_tick(self):
        harness = CircuitTestHarness.from_ascii(". c w -")
        harness.tick()
        self.assertEqual(harness.read_net((3, 0)), 1)
        harness.tick()
        self.assertEqual(harness.read_net((3, 0)), 0)
        harness.tick()
        self.assertEqual(harness.read_net((3, 0)), 1)

    def test_clock_uses_project_interval(self):
        harness = CircuitTestHarness.from_ascii(". c w -", clock_interval=3)

        observed = []
        for _ in range(7):
            harness.tick()
            observed.append(harness.read_net((3, 0)))

        self.assertEqual(observed, [1, 1, 1, 0, 0, 0, 1])

    def test_led_turns_on_when_any_read_input_is_on_and_has_no_outputs(self):
        harness = CircuitTestHarness.from_ascii(
            ". . - . .",
            ". . r . .",
            "- r d r -",
            ". . - . .",
        )
        led = harness.gate_at((2, 2))
        self.assertEqual(led.kind, "LED")
        self.assertEqual(len(led.inputs), 3)
        self.assertEqual(led.outputs, [])

        trace = []
        trace.append(harness.read_gate((2, 2)))
        harness.drive({(2, 0): 0, (0, 2): 0, (4, 2): 0}).tick()
        trace.append(harness.read_gate((2, 2)))
        harness.drive({(2, 0): 1, (0, 2): 0, (4, 2): 0}).tick()
        trace.append(harness.read_gate((2, 2)))
        harness.drive({(2, 0): 0, (0, 2): 0, (4, 2): 0}).tick()
        trace.append(harness.read_gate((2, 2)))
        self.assertEqual(trace, [0, 0, 1, 0])

    def test_random_block_uses_deterministic_xorshift_sequence(self):
        harness = CircuitTestHarness.from_ascii(". ? w -")
        seed = (0x9E3779B9 ^ 1) & 0xFFFFFFFF
        expected = []
        for _ in range(5):
            seed ^= (seed << 13) & 0xFFFFFFFF
            seed ^= (seed >> 17) & 0xFFFFFFFF
            seed ^= (seed << 5) & 0xFFFFFFFF
            seed &= 0xFFFFFFFF
            expected.append(seed & 1)

        observed = []
        for _ in range(5):
            harness.tick()
            observed.append(harness.read_net((3, 0)))
        self.assertEqual(observed, expected)

    def test_multiple_writers_reduce_with_wired_or(self):
        harness = CircuitTestHarness.from_ascii("- r b w - w b r -")
        harness.drive({(0, 0): 1, (8, 0): 0}).tick()
        self.assertEqual(harness.read_net((4, 0)), 1)

        harness.initialize()
        harness.drive({(0, 0): 0, (8, 0): 1}).tick()
        self.assertEqual(harness.read_net((4, 0)), 1)

        harness.initialize()
        harness.drive({(0, 0): 0, (8, 0): 0}).tick()
        self.assertEqual(harness.read_net((4, 0)), 0)

    def test_crossroads_keep_crossing_nets_separate(self):
        harness = CircuitTestHarness.from_ascii(
            ". . - . .",
            ". . - . .",
            "- - + - -",
            ". . - . .",
            ". . - . .",
        )
        self.assertEqual(len(harness.nets), 2)
        self.assertEqual(harness.net_at((2, 0)).id, harness.net_at((2, 4)).id)
        self.assertEqual(harness.net_at((0, 2)).id, harness.net_at((4, 2)).id)
        self.assertNotEqual(harness.net_at((2, 0)).id, harness.net_at((0, 2)).id)

    def test_crossroads_connect_bus_pairs_without_merging_axes(self):
        harness = CircuitTestHarness.from_ascii(
            ". . - . .",
            ". . = . .",
            "~ = + = ~",
            ". . = . .",
            ". . - . .",
            token_map={"~": BLOCKS["WIRE_1"]},
        )
        self.assertEqual(len(harness.nets), 2)
        self.assertEqual(len(harness.compiler.bus_components), 2)
        self.assertEqual(harness.net_at((2, 0)).id, harness.net_at((2, 4)).id)
        self.assertEqual(harness.net_at((0, 2)).id, harness.net_at((4, 2)).id)
        self.assertNotEqual(harness.net_at((2, 0)).id, harness.net_at((0, 2)).id)

        visual_blocks = {(block.x, block.y): block for block in harness.compiler.build_visual_blocks()}
        self.assertEqual(visual_blocks[(2, 2)].signal_kind, harness.compiler.VISUAL_SIGNAL_NONE)

    def test_bus_contacts_can_reach_trace_through_crossroad(self):
        harness = CircuitTestHarness.from_ascii("- + = = + -")
        self.assertEqual(len(harness.nets), 1)
        self.assertEqual(harness.net_at((0, 0)).id, harness.net_at((5, 0)).id)

    def test_bus_segments_connect_through_matching_tunnels(self):
        harness = CircuitTestHarness.from_ascii("- = t . . t = -")
        self.assertEqual(len(harness.compiler.bus_components), 1)
        self.assertEqual(len(harness.nets), 1)
        self.assertEqual(harness.net_at((0, 0)).id, harness.net_at((7, 0)).id)

        harness.drive({(0, 0): 1})
        self.assertEqual(harness.read_net((7, 0)), 1)

    def test_overlapping_bus_tunnels_match_by_bus_color(self):
        harness = CircuitTestHarness.from_ascii(
            "- = t . ~ ! t . . t ! ~ . t = -",
            token_map={"~": BLOCKS["WIRE_1"], "!": BLOCKS["BUS_1"]},
        )
        self.assertEqual(len(harness.compiler.bus_components), 2)
        self.assertEqual(len(harness.nets), 2)
        self.assertEqual(harness.net_at((0, 0)).id, harness.net_at((15, 0)).id)
        self.assertEqual(harness.net_at((4, 0)).id, harness.net_at((11, 0)).id)
        self.assertNotEqual(harness.net_at((0, 0)).id, harness.net_at((4, 0)).id)

    def test_tunnels_connect_separated_wire_segments(self):
        harness = CircuitTestHarness.from_ascii(". - t . . t -")
        self.assertEqual(len(harness.nets), 1)
        self.assertEqual(harness.net_at((1, 0)).id, harness.net_at((6, 0)).id)

    def test_overlapping_tunnels_match_by_adjacent_wire_color(self):
        harness = CircuitTestHarness.from_ascii(
            "- t . ~ t . . t ~ . t -",
            token_map={"~": BLOCKS["WIRE_1"]},
        )
        self.assertEqual(len(harness.nets), 2)
        self.assertEqual(harness.net_at((0, 0)).id, harness.net_at((11, 0)).id)
        self.assertEqual(harness.net_at((3, 0)).id, harness.net_at((8, 0)).id)
        self.assertNotEqual(harness.net_at((0, 0)).id, harness.net_at((3, 0)).id)

    def test_bus_merges_same_lane_but_keeps_other_lanes_separate(self):
        harness = CircuitTestHarness.from_ascii(
            "- = ! = -",
            ". = ! = .",
            "~ = ! = ~",
            token_map={"~": BLOCKS["WIRE_1"], "!": BLOCKS["BUS_1"]},
        )
        self.assertEqual(len(harness.nets), 2)
        self.assertEqual(harness.net_at((0, 0)).id, harness.net_at((4, 0)).id)
        self.assertEqual(harness.net_at((0, 2)).id, harness.net_at((4, 2)).id)
        self.assertNotEqual(harness.net_at((0, 0)).id, harness.net_at((0, 2)).id)

        harness.drive({(0, 0): 1, (0, 2): 0})
        self.assertEqual(harness.read_net((4, 0)), 1)
        self.assertEqual(harness.read_net((4, 2)), 0)

    def test_bus_highlight_tracks_any_attached_lane(self):
        harness = CircuitTestHarness.from_ascii(
            "- = ! = -",
            ". = ! = .",
            "~ = ! = ~",
            token_map={"~": BLOCKS["WIRE_1"], "!": BLOCKS["BUS_1"]},
        )
        visual_blocks = {(block.x, block.y): block for block in harness.compiler.build_visual_blocks()}
        bus_block = visual_blocks[(1, 1)]
        self.assertEqual(bus_block.signal_kind, harness.compiler.VISUAL_SIGNAL_BUS)
        self.assertEqual(bus_block.signal_id, 1)

    def test_tunnels_require_a_matching_far_side_trace_block(self):
        with self.assertRaisesRegex(ValueError, "could not find a matching tunnel exit"):
            CircuitTestHarness.from_ascii(
                "- t . . t = -",
            )

    def test_touching_wire_colors_still_form_one_net(self):
        harness = CircuitTestHarness.from_ascii(
            "- ~ -",
            token_map={"~": BLOCKS["WIRE_1"]},
        )
        self.assertEqual(len(harness.nets), 1)
        self.assertEqual(harness.net_at((0, 0)).id, harness.net_at((2, 0)).id)

    def test_same_gate_kind_can_merge_through_bus(self):
        harness = CircuitTestHarness.from_ascii(
            "a = ! = a",
            token_map={"!": BLOCKS["BUS_1"]},
        )
        self.assertEqual(len(harness.gates), 1)
        self.assertEqual(harness.gate_at((0, 0)).id, harness.gate_at((4, 0)).id)

    def test_same_gate_kind_can_merge_through_tunnels(self):
        harness = CircuitTestHarness.from_ascii("- r b t . . t b w -")
        self.assertEqual(len(harness.gates), 1)
        self.assertEqual(harness.gate_at((2, 0)).id, harness.gate_at((7, 0)).id)

        harness.drive({(0, 0): 1}).tick()
        self.assertEqual(harness.read_net((9, 0)), 1)

    def test_vmem_virtual_blocks_render_on_empty_cells(self):
        harness = simple_vmem_harness()
        visual_blocks = {(block.x, block.y): block for block in harness.compiler.build_visual_blocks()}

        address_block = visual_blocks[(2, 1)]
        data_block = visual_blocks[(2, 3)]
        self.assertEqual(address_block.signal_kind, harness.compiler.VISUAL_SIGNAL_GATE)
        self.assertEqual(data_block.signal_kind, harness.compiler.VISUAL_SIGNAL_GATE)
        self.assertEqual(address_block.style_flags, harness.compiler.VISUAL_STYLE_VMEM)
        self.assertEqual(data_block.style_flags, harness.compiler.VISUAL_STYLE_VMEM)
        self.assertIsNotNone(address_block.base_color)
        self.assertIsNotNone(data_block.base_color)

    def test_vmem_loads_the_new_word_when_address_changes(self):
        harness = simple_vmem_harness({1: 1})
        self.assertEqual(harness.read_gate((2, 1)), 0)
        self.assertEqual(harness.read_gate((2, 3)), 0)
        self.assertEqual(harness.read_net((4, 3)), 0)

        harness.drive({(0, 1): 1}).tick()
        self.assertEqual(harness.read_gate((2, 1)), 1)
        self.assertEqual(harness.read_gate((2, 3)), 1)
        self.assertEqual(harness.read_net((4, 3)), 0)
        self.assertTrue(harness.vmem_lock_pending)

        harness.drive({(0, 1): 0}).tick()
        self.assertEqual(harness.read_gate((2, 1)), 1)
        self.assertEqual(harness.read_net((4, 3)), 1)
        self.assertFalse(harness.vmem_lock_pending)

    def test_vmem_loaded_data_can_drive_latch_on_the_locked_tick(self):
        harness = vmem_data_latch_harness({1: 1})
        self.assertEqual(harness.read_net((8, 3)), 0)

        harness.drive({(0, 1): 1}).tick()
        self.assertEqual(harness.read_gate((2, 3)), 1)
        self.assertEqual(harness.read_net((8, 3)), 0)

        harness.drive({(0, 1): 0}).tick()
        self.assertEqual(harness.read_net((8, 3)), 0)

        harness.tick()
        self.assertEqual(harness.read_net((8, 3)), 1)

    def test_vmem_address_latch_cancels_even_simultaneous_rising_inputs(self):
        harness, input_points = multi_read_vmem_latch_harness("address", 2)

        harness.drive({point: 1 for point in input_points}).tick()

        self.assertEqual(harness.vmem_address_value, 0)
        self.assertEqual(harness.read_gate((2, 1)), 0)
        self.assertFalse(harness.vmem_lock_pending)

    def test_vmem_address_latch_flips_on_odd_simultaneous_rising_inputs(self):
        harness, input_points = multi_read_vmem_latch_harness("address", 3)

        harness.drive({point: 1 for point in input_points}).tick()

        self.assertEqual(harness.vmem_address_value, 1)
        self.assertEqual(harness.read_gate((2, 2)), 1)
        self.assertTrue(harness.vmem_lock_pending)

    def test_vmem_data_latch_cancels_even_simultaneous_rising_inputs(self):
        harness, input_points = multi_read_vmem_latch_harness("data", 2)

        harness.drive({point: 1 for point in input_points}).tick()

        self.assertEqual(harness.vmem_data_value, 0)
        self.assertEqual(harness.read_gate((2, 1)), 0)
        self.assertEqual(harness.read_vmem_word(0), 0)

    def test_vmem_data_latch_flips_on_odd_simultaneous_rising_inputs(self):
        harness, input_points = multi_read_vmem_latch_harness("data", 3)

        harness.drive({point: 1 for point in input_points}).tick()

        self.assertEqual(harness.vmem_data_value, 1)
        self.assertEqual(harness.read_gate((2, 2)), 1)
        self.assertEqual(harness.read_vmem_word(0), 1)

    def test_vmem_rejects_address_toggles_while_locked(self):
        vmem_config = VMemConfig(
            enabled=True,
            address_layout=VMemLayout(bits=2, position_x=2, position_y=1, offset_x=0, offset_y=2, size_x=1, size_y=1),
            data_layout=VMemLayout(bits=1, position_x=2, position_y=5, offset_x=0, offset_y=0, size_x=1, size_y=1),
        )
        harness = CircuitTestHarness.from_ascii(
            ". . . . .",
            "- r . w -",
            ". . . . .",
            "- r . w -",
            ". . . . .",
            "- r . w -",
            vmem_config=vmem_config,
            vmem_words={1: 1},
        )

        harness.drive({(0, 1): 1}).tick()
        self.assertTrue(harness.vmem_lock_pending)

        with self.assertRaisesRegex(RuntimeError, "address latches cannot be changed"):
            harness.drive({(0, 1): 0, (0, 3): 1}).tick()

    def test_vmem_rejects_data_toggles_while_locked_then_unlocks(self):
        harness = simple_vmem_harness({1: 1})
        harness.drive({(0, 1): 1}).tick()

        with self.assertRaisesRegex(RuntimeError, "data latches cannot be changed"):
            harness.drive({(0, 1): 0, (0, 3): 1}).tick()

        harness = simple_vmem_harness({1: 1})
        harness.drive({(0, 1): 1}).tick()
        harness.drive({(0, 1): 0}).tick()
        harness.drive({(0, 3): 1}).tick()
        self.assertEqual(harness.read_gate((2, 3)), 0)

    def test_gpu7_vmem_lsb_address_loads_word_one_data(self):
        project_path = Path.joinpath(VCB_SIMULATOR_ROOT, "fixtures", "gpu7.vcb")
        project_data = load_project(project_path)
        harness = CircuitTestHarness(
            load_grid_from_project_data(project_data),
            vmem_config=build_vmem_config(project_data, project_path),
            clock_interval=int(project_data.get("clock_interval", 1)),
        )
        address_lsb = next(gate for gate in harness.gates.values() if gate.kind == VMEM_ADDRESS_KIND and gate.vmem_value_bit == 0)
        data_bit_19 = next(gate for gate in harness.gates.values() if gate.kind == VMEM_DATA_KIND and gate.vmem_value_bit == 19)

        harness.net_state[address_lsb.inputs[0].id] = 1
        harness.tick()

        self.assertEqual(harness.vmem_address_value, 0x00001)
        self.assertEqual(harness.vmem_loaded_address, 0x00001)
        self.assertEqual(harness.vmem_data_value, 0x00080000)
        self.assertEqual(harness.gate_out[data_bit_19.id], 1)

    def test_vmem_writes_current_word_before_loading_new_address(self):
        harness = simple_vmem_harness({1: 1})
        harness.drive({(0, 1): 1, (0, 3): 1}).tick()

        self.assertEqual(harness.read_vmem_word(1), 1)
        self.assertEqual(harness.read_net((4, 3)), 1)

        harness.drive({(0, 1): 0, (0, 3): 0}).tick()
        self.assertEqual(harness.read_net((4, 3)), 1)

        harness.drive({(0, 1): 1, (0, 3): 0}).tick()
        harness.drive({(0, 1): 0, (0, 3): 0}).tick()
        self.assertEqual(harness.read_net((4, 3)), 1)

    def test_vmem_uses_reversed_bit_order_with_inverted_x_offset(self):
        harness = CircuitTestHarness.from_ascii(
            ". . . . . .",
            ". . . . . .",
            ". . . . . .",
            ". . . . . .",
            ". . . . . .",
            ". . . . . .",
            vmem_config=VMemConfig(
                enabled=True,
                address_layout=VMemLayout(
                    bits=2,
                    position_x=2,
                    position_y=2,
                    offset_x=1,
                    offset_y=-1,
                    size_x=1,
                    size_y=1,
                ),
                data_layout=VMemLayout(
                    bits=1,
                    position_x=5,
                    position_y=5,
                    offset_x=0,
                    offset_y=0,
                    size_x=1,
                    size_y=1,
                ),
            ),
        )

        address_gate_blocks = {next(iter(gate.blocks)) for gate in harness.compiler.gates.values() if gate.kind == "VMEM_ADDRESS"}
        self.assertIn((1, 1), address_gate_blocks)
        self.assertIn((2, 2), address_gate_blocks)
        self.assertNotIn((3, 1), address_gate_blocks)
        self.assertNotIn((3, 3), address_gate_blocks)


if __name__ == "__main__":
    unittest.main()
