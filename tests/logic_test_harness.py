from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from compiler.compiler import VMEM_ADDRESS_KIND, VMEM_DATA_KIND, CircuitCompiler, Gate, Net, VMemConfig
from main import BLOCKS, build_config

Grid = List[List[int]]
Point = Tuple[int, int]


DEFAULT_TOKEN_MAP: Dict[str, int] = {
    ".": 0,
    "w": BLOCKS["WRITE"],
    "r": BLOCKS["READ"],
    "-": BLOCKS["WIRE_0"],
    "=": BLOCKS["BUS_0"],
    "+": BLOCKS["CROSS"],
    "t": BLOCKS["TUNNEL"],
    "b": BLOCKS["BUFFER"],
    "d": BLOCKS["LED"],
    "a": BLOCKS["AND"],
    "o": BLOCKS["OR"],
    "x": BLOCKS["XOR"],
    "n": BLOCKS["NOT"],
    "q": BLOCKS["NOR"],
    "l": BLOCKS["LATCH_OFF"],
    "L": BLOCKS["LATCH_ON"],
    "c": BLOCKS["CLOCK"],
    "?": BLOCKS["RANDOM"],
    "p": BLOCKS["BREAKPOINT"],
}


@dataclass
class Probe:
    net_id: int
    point: Point


def grid_from_ascii(
    *rows: str,
    token_map: Optional[Dict[str, int]] = None,
) -> Grid:
    """Build a small test grid from space-separated ASCII tokens."""
    if not rows:
        raise ValueError("At least one row is required.")

    resolved_token_map = dict(DEFAULT_TOKEN_MAP)
    if token_map is not None:
        resolved_token_map.update(token_map)

    parsed_rows = [row.split() for row in rows]
    width = len(parsed_rows[0])
    if width == 0:
        raise ValueError("Rows must not be empty.")

    for row in parsed_rows:
        if len(row) != width:
            raise ValueError("All rows must have the same token count.")

    grid: Grid = []
    for row in parsed_rows:
        parsed_row: List[int] = []
        for token in row:
            if token not in resolved_token_map:
                raise KeyError(f"Unknown token '{token}'.")
            parsed_row.append(resolved_token_map[token])
        grid.append(parsed_row)
    return grid


class CircuitTestHarness:
    """
    Lightweight Python-side simulator for compiled VCB circuits.

    This intentionally mirrors the emitted vcb_simulator semantics closely enough for
    fast regression tests:
    - gates read the current net state
    - nets are updated after all gate outputs are computed
    - writer-less nets retain their previous value, which makes them useful as
      stable external inputs in tests
    """

    def __init__(
        self,
        grid: Grid,
        vmem_config: Optional[VMemConfig] = None,
        clock_interval: int = 1,
        vmem_words: Optional[Dict[int, int]] = None,
    ):
        self.grid = grid
        self.clock_interval = max(1, int(clock_interval))
        self.compiler = CircuitCompiler(
            grid,
            build_config(),
            vmem_config=vmem_config,
            clock_interval=self.clock_interval,
        )
        self.gates, self.nets = self.compiler.compile()
        self.regions = self.compiler.build_regions()
        self.vmem_config = vmem_config
        self.gate_out = [0 for _ in range(self._gate_count() + 1)]
        self.latch_state = [0 for _ in range(self._gate_count() + 1)]
        self.latch_input_state: Dict[int, List[int]] = {}
        self.rng_state = [0 for _ in range(self._gate_count() + 1)]
        self.net_state = [0 for _ in range(self._net_count() + 1)]
        self.tick_count = 0
        self.vmem_project_words: Dict[int, int] = {address: value & 0xFFFFFFFF for address, value in (vmem_words or {}).items()}
        self.vmem_words: Dict[int, int] = {}
        self.vmem_address_value = 0
        self.vmem_loaded_address = 0
        self.vmem_pending_address = 0
        self.vmem_data_value = 0
        self.vmem_lock_pending = False
        self.breakpoint_was_active = False
        self.initialize()

    @classmethod
    def from_ascii(
        cls,
        *rows: str,
        token_map: Optional[Dict[str, int]] = None,
        vmem_config: Optional[VMemConfig] = None,
        clock_interval: int = 1,
        vmem_words: Optional[Dict[int, int]] = None,
    ) -> "CircuitTestHarness":
        return cls(
            grid_from_ascii(*rows, token_map=token_map),
            vmem_config=vmem_config,
            clock_interval=clock_interval,
            vmem_words=vmem_words,
        )

    def initialize(self) -> None:
        self.gate_out = [0 for _ in range(self._gate_count() + 1)]
        self.latch_state = [0 for _ in range(self._gate_count() + 1)]
        self.latch_input_state = {}
        self.rng_state = [0 for _ in range(self._gate_count() + 1)]
        self.net_state = [0 for _ in range(self._net_count() + 1)]
        self.tick_count = 0
        self.vmem_address_value = 0
        self.vmem_loaded_address = 0
        self.vmem_pending_address = 0
        self.vmem_data_value = 0
        self.vmem_lock_pending = False
        self.breakpoint_was_active = False

        self._initialize_vmem_words()

        for gate in self.gates.values():
            if gate.kind == "LATCH":
                self.gate_out[gate.id] = gate.initial_state
                self.latch_state[gate.id] = gate.initial_state
                self.latch_input_state[gate.id] = [0 for _ in gate.inputs]
            if gate.kind == "RANDOM":
                self.rng_state[gate.id] = self._random_seed(gate.id)
            if gate.kind in {VMEM_ADDRESS_KIND, VMEM_DATA_KIND}:
                self.latch_input_state[gate.id] = [0 for _ in gate.inputs]

        self._apply_vmem_gate_outputs()

    def tick(self, count: int = 1) -> "CircuitTestHarness":
        for _ in range(count):
            vmem_locked_this_tick = self.vmem_lock_pending
            vmem_address_changed = False
            vmem_data_changed = False
            for region in self.regions:
                for gate in region.gates:
                    changed = self._update_gate(gate, vmem_locked_this_tick)
                    if gate.kind == VMEM_ADDRESS_KIND:
                        vmem_address_changed |= changed
                    elif gate.kind == VMEM_DATA_KIND:
                        vmem_data_changed |= changed
            next_net_state = [0 for _ in range(self._net_count() + 1)]
            for net in self._sorted_nets():
                if net.writers:
                    next_net_state[net.id] = self._reduce(
                        (self.gate_out[gate.id] for gate in net.writers),
                        "or",
                        0,
                    )
                else:
                    next_net_state[net.id] = self.net_state[net.id]
            if self._vmem_enabled():
                if vmem_data_changed:
                    self._write_loaded_vmem_word(self.vmem_data_value)
                if vmem_locked_this_tick:
                    self.vmem_lock_pending = False
                elif vmem_address_changed:
                    self.vmem_pending_address = self.vmem_address_value & self.vmem_config.address_mask
                    self._load_pending_vmem_word()
                    self.vmem_lock_pending = True
            self.net_state = next_net_state
            self.tick_count += 1
        return self

    def set_net(self, point: Point, state: int) -> int:
        net = self.net_at(point)
        self.net_state[net.id] = 1 if state else 0
        return net.id

    def drive(self, mapping: Dict[Point, int]) -> "CircuitTestHarness":
        for point, state in mapping.items():
            self.set_net(point, state)
        return self

    def read_net(self, point: Point) -> int:
        return self.net_state[self.net_at(point).id]

    def set_latch(self, point: Point, state: int) -> int:
        gate = self.gate_at(point)
        if gate.kind != "LATCH":
            raise ValueError(f"Gate at {point} is {gate.kind}, not a latch.")
        bit = 1 if state else 0
        self.latch_state[gate.id] = bit
        self.gate_out[gate.id] = bit
        return gate.id

    def read_gate(self, point: Point) -> int:
        return self.gate_out[self.gate_at(point).id]

    def breakpoint_active(self) -> bool:
        active = any(self.gate_out[gate.id] != 0 for gate in self.gates.values() if gate.kind == "BREAKPOINT")
        if not active:
            self.breakpoint_was_active = False
            return False
        if self.breakpoint_was_active:
            return False
        self.breakpoint_was_active = True
        return True

    def set_vmem_word(self, address: int, value: int) -> None:
        if not self._vmem_enabled():
            raise ValueError("VMem is not enabled for this harness.")
        bounded_address = address & self.vmem_config.address_mask
        bounded_value = value & 0xFFFFFFFF
        self.vmem_project_words[bounded_address] = bounded_value
        self.vmem_words[bounded_address] = bounded_value
        if self.vmem_loaded_address == bounded_address:
            self.vmem_data_value = bounded_value & self.vmem_config.data_mask
            self._apply_vmem_data_gate_outputs()

    def read_vmem_word(self, address: int) -> int:
        if not self._vmem_enabled():
            raise ValueError("VMem is not enabled for this harness.")
        bounded_address = address & self.vmem_config.address_mask
        return self.vmem_words.get(bounded_address, 0)

    def gate_at(self, point: Point) -> Gate:
        gate = self.compiler.coord_to_gate.get(point)
        if gate is None:
            raise KeyError(f"No gate found at {point}.")
        return gate

    def net_at(self, point: Point) -> Net:
        net = self.compiler.coord_to_net.get(point)
        if net is None:
            raise KeyError(f"No net found at {point}.")
        return net

    def probes_for(self, points: Iterable[Point]) -> List[Probe]:
        return [Probe(net_id=self.net_at(point).id, point=point) for point in points]

    def trace(
        self,
        input_steps: Sequence[Dict[Point, int]],
        probe_points: Sequence[Point],
        reset: bool = True,
        include_initial: bool = False,
    ) -> List[Dict[Point, int]]:
        if reset:
            self.initialize()
        snapshots: List[Dict[Point, int]] = []
        if include_initial:
            snapshots.append({point: self.read_net(point) for point in probe_points})
        for step in input_steps:
            self.drive(step)
            self.tick()
            snapshots.append({point: self.read_net(point) for point in probe_points})
        return snapshots

    def trace_values(
        self,
        input_steps: Sequence[Dict[Point, int]],
        probe_point: Point,
        reset: bool = True,
        include_initial: bool = False,
    ) -> List[int]:
        snapshots = self.trace(
            input_steps,
            [probe_point],
            reset=reset,
            include_initial=include_initial,
        )
        return [snapshot[probe_point] for snapshot in snapshots]

    def truth_table(
        self,
        input_points: Sequence[Point],
        output_point: Point,
        settle_ticks: int = 1,
        reset_between_cases: bool = True,
    ) -> Dict[Tuple[int, ...], int]:
        table: Dict[Tuple[int, ...], int] = {}
        for bits in product((0, 1), repeat=len(input_points)):
            if reset_between_cases:
                self.initialize()
            self.drive({point: bit for point, bit in zip(input_points, bits)})
            self.tick(settle_ticks)
            table[bits] = self.read_net(output_point)
        return table

    def _update_gate(self, gate: Gate, vmem_locked_this_tick: bool = False) -> bool:
        input_values = [self.net_state[net.id] for net in gate.inputs]
        if gate.kind == "BUFFER":
            self.gate_out[gate.id] = self._reduce(input_values, "or", 0)
            return False
        if gate.kind == "LED":
            self.gate_out[gate.id] = self._reduce(input_values, "or", 0)
            return False
        if gate.kind == "BREAKPOINT":
            self.gate_out[gate.id] = self._reduce(input_values, "or", 0)
            return False
        if gate.kind == "AND":
            self.gate_out[gate.id] = self._reduce(input_values, "and", 1 if input_values else 0)
            return False
        if gate.kind == "OR":
            self.gate_out[gate.id] = self._reduce(input_values, "or", 0)
            return False
        if gate.kind == "XOR":
            self.gate_out[gate.id] = self._reduce(input_values, "xor", 0)
            return False
        if gate.kind in {"NOT", "NOR"}:
            self.gate_out[gate.id] = 0 if self._reduce(input_values, "or", 0) else 1
            return False
        if gate.kind == "CLOCK":
            if self._clock_should_toggle():
                self.gate_out[gate.id] ^= 1
            return False
        if gate.kind == "RANDOM":
            self.rng_state[gate.id] = self._next_rng(self.rng_state[gate.id])
            enabled = self._reduce(input_values, "or", 1 if not input_values else 0)
            self.gate_out[gate.id] = (self.rng_state[gate.id] & 1) if enabled else 0
            return False
        if gate.kind == "LATCH":
            previous_values = self.latch_input_state[gate.id]
            rising_edge_detected = any(current_value and not previous_value for current_value, previous_value in zip(input_values, previous_values))
            if rising_edge_detected:
                self.latch_state[gate.id] ^= 1
            self.latch_input_state[gate.id] = [1 if value else 0 for value in input_values]
            self.gate_out[gate.id] = self.latch_state[gate.id]
            return rising_edge_detected
        if gate.kind == VMEM_ADDRESS_KIND:
            return self._update_vmem_address_gate(gate, input_values, vmem_locked_this_tick)
        if gate.kind == VMEM_DATA_KIND:
            return self._update_vmem_data_gate(gate, input_values, vmem_locked_this_tick)
        raise ValueError(f"Unsupported gate kind: {gate.kind}")

    def _update_vmem_address_gate(
        self,
        gate: Gate,
        input_values: List[int],
        vmem_locked_this_tick: bool,
    ) -> bool:
        previous_values = self.latch_input_state[gate.id]
        rising_edge_count = sum(1 for current_value, previous_value in zip(input_values, previous_values) if current_value and not previous_value)
        state_change_detected = (rising_edge_count % 2) == 1
        if state_change_detected and vmem_locked_this_tick:
            raise RuntimeError("VMem address latches cannot be changed while VMEM is locked.")
        if state_change_detected and gate.vmem_value_bit is not None:
            self.vmem_address_value ^= 1 << gate.vmem_value_bit
        self.latch_input_state[gate.id] = [1 if value else 0 for value in input_values]
        self.gate_out[gate.id] = self._bit_value(self.vmem_address_value, gate.vmem_value_bit)
        return state_change_detected and not vmem_locked_this_tick

    def _update_vmem_data_gate(
        self,
        gate: Gate,
        input_values: List[int],
        vmem_locked_this_tick: bool,
    ) -> bool:
        previous_values = self.latch_input_state[gate.id]
        rising_edge_count = sum(1 for current_value, previous_value in zip(input_values, previous_values) if current_value and not previous_value)
        state_change_detected = (rising_edge_count % 2) == 1
        if state_change_detected and vmem_locked_this_tick:
            raise RuntimeError("VMem data latches cannot be changed while VMEM is locked.")
        if state_change_detected and gate.vmem_value_bit is not None:
            self.vmem_data_value ^= 1 << gate.vmem_value_bit
        self.latch_input_state[gate.id] = [1 if value else 0 for value in input_values]
        self.gate_out[gate.id] = self._bit_value(self.vmem_data_value, gate.vmem_value_bit)
        return state_change_detected and not vmem_locked_this_tick

    def _sorted_nets(self) -> List[Net]:
        return [self.nets[net_id] for net_id in sorted(self.nets)]

    def _gate_count(self) -> int:
        return max(self.gates, default=0)

    def _net_count(self) -> int:
        return max(self.nets, default=0)

    @staticmethod
    def _reduce(values: Iterable[int], operator: str, empty_value: int) -> int:
        iterator = iter(values)
        try:
            result = next(iterator)
        except StopIteration:
            return empty_value

        for value in iterator:
            if operator == "and":
                result &= value
            elif operator == "or":
                result |= value
            elif operator == "xor":
                result ^= value
            else:
                raise ValueError(f"Unsupported reduction operator: {operator}")
        return 1 if result else 0

    @staticmethod
    def _random_seed(gate_id: int) -> int:
        seed = (0x9E3779B9 ^ gate_id) & 0xFFFFFFFF
        return seed or 1

    @staticmethod
    def _next_rng(state: int) -> int:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17) & 0xFFFFFFFF
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    @staticmethod
    def _bit_value(value: int, bit_index: Optional[int]) -> int:
        if bit_index is None:
            return 0
        return 1 if ((value >> bit_index) & 1) else 0

    def _clock_should_toggle(self) -> bool:
        return self.tick_count == 0 or (self.tick_count % self.clock_interval) == 0

    def _vmem_enabled(self) -> bool:
        return self.vmem_config is not None and self.vmem_config.enabled

    def _initialize_vmem_words(self) -> None:
        if not self._vmem_enabled():
            self.vmem_project_words = {}
            self.vmem_words = {}
            return

        if self.vmem_project_words:
            self._apply_persistent_vmem_range()
        else:
            self.vmem_project_words = self._decode_vmem_words()
        self.vmem_words = dict(self.vmem_project_words)
        self.vmem_address_value = 0
        self.vmem_loaded_address = 0
        self.vmem_pending_address = 0
        self.vmem_data_value = self.read_vmem_word(0) & self.vmem_config.data_mask

    def _apply_vmem_gate_outputs(self) -> None:
        if not self._vmem_enabled():
            return
        self._apply_vmem_address_gate_outputs()
        self._apply_vmem_data_gate_outputs()

    def _apply_vmem_address_gate_outputs(self) -> None:
        for gate in self.gates.values():
            if gate.kind != VMEM_ADDRESS_KIND:
                continue
            self.gate_out[gate.id] = self._bit_value(self.vmem_address_value, gate.vmem_value_bit)

    def _apply_vmem_data_gate_outputs(self) -> None:
        for gate in self.gates.values():
            if gate.kind != VMEM_DATA_KIND:
                continue
            self.gate_out[gate.id] = self._bit_value(self.vmem_data_value, gate.vmem_value_bit)

    def _load_pending_vmem_word(self) -> None:
        self.vmem_loaded_address = self.vmem_pending_address & self.vmem_config.address_mask
        self.vmem_data_value = self.read_vmem_word(self.vmem_loaded_address) & self.vmem_config.data_mask
        self.vmem_lock_pending = False
        self._apply_vmem_data_gate_outputs()

    def _reduce_vmem_data_output_nets(self) -> None:
        output_net_ids = {net.id for gate in self.gates.values() if gate.kind == VMEM_DATA_KIND for net in gate.outputs}
        for net_id in output_net_ids:
            net = self.nets[net_id]
            self.net_state[net_id] = self._reduce(
                (self.gate_out[gate.id] for gate in net.writers),
                "or",
                0,
            )

    def _write_loaded_vmem_word(self, value: int) -> None:
        if not self._vmem_enabled():
            return
        current_word = self.vmem_words.get(self.vmem_loaded_address, 0)
        preserved = current_word & ~self.vmem_config.data_mask
        self.vmem_words[self.vmem_loaded_address] = preserved | (value & self.vmem_config.data_mask)

    def _decode_vmem_words(self) -> Dict[int, int]:
        if not self._vmem_enabled() or not self.vmem_config.initial_data_be:
            return {}

        raw = self.vmem_config.initial_data_be
        words: Dict[int, int] = {}
        for index in range(0, len(raw), 4):
            value = int.from_bytes(raw[index : index + 4], byteorder="big", signed=False)
            if value != 0:
                words[index // 4] = value
        return words

    def _apply_persistent_vmem_range(self) -> None:
        if not self._vmem_enabled():
            return
        start = self.vmem_config.persistent_start
        end = self.vmem_config.persistent_end
        if start > end:
            return
        for address in range(start, end + 1):
            value = self.vmem_words.get(address, 0)
            if value == 0:
                self.vmem_project_words.pop(address, None)
            else:
                self.vmem_project_words[address] = value
