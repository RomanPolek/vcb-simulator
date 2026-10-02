from __future__ import annotations

import sys
import threading
import time
from array import array
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from compiler.compiler import (
    VMEM_ADDRESS_KIND,
    VMEM_DATA_KIND,
    VisualBlock,
)
from runner.context import EmitterContext

BASIC_KINDS = {"AND", "OR", "XOR", "NOT", "BUFFER", "NOR"}
KIND_AND = 1
KIND_OR = 2
KIND_XOR = 3
KIND_NOT = 4
KIND_BUFFER = 5
KIND_NOR = 6
KIND_LED = 7
KIND_BREAKPOINT = 8
KIND_LATCH = 9
KIND_RANDOM = 10
KIND_VMEM_ADDRESS = 11
KIND_VMEM_DATA = 12

KIND_CODES = {
    "AND": KIND_AND,
    "OR": KIND_OR,
    "XOR": KIND_XOR,
    "NOT": KIND_NOT,
    "BUFFER": KIND_BUFFER,
    "NOR": KIND_NOR,
    "LED": KIND_LED,
    "BREAKPOINT": KIND_BREAKPOINT,
    "LATCH": KIND_LATCH,
    "RANDOM": KIND_RANDOM,
    VMEM_ADDRESS_KIND: KIND_VMEM_ADDRESS,
    VMEM_DATA_KIND: KIND_VMEM_DATA,
}
OR_LIKE_CODES = frozenset((KIND_BUFFER, KIND_OR, KIND_LED, KIND_BREAKPOINT))
NOT_LIKE_CODES = frozenset((KIND_NOT, KIND_NOR))


@dataclass(slots=True)
class RuntimeOp:
    gate_id: int
    kind: str
    kind_code: int
    input_net_ids: Tuple[int, ...]
    output_net_ids: Tuple[int, ...]
    initial_state: int = 0
    value_bit: int | None = None
    input_mask: int = 0


@dataclass(frozen=True, slots=True)
class VMemSnapshot:
    enabled: bool
    word_count: int
    address_bits: int
    address_mask: int
    data_bits: int
    data_mask: int
    address_value: int
    active_address: int
    data_value: int
    lock_pending: bool


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    tick: int
    gate_out: bytes
    net_state: bytes
    bus_state: bytes
    vmem: VMemSnapshot
    vdisplay_words: Tuple[int, ...]


class CircuitRuntime:
    """
    Execute a compiled VCB circuit directly in Python.

    The runtime keeps the generated C emitter's important timing model:
    gate outputs write into the next net state, changed nets schedule their
    readers for the next tick, and VMEM data publishes on the following tick.
    It deliberately skips the generated tables' packed masks and locality tricks.
    """

    def __init__(self, context: EmitterContext):
        self.ctx = context
        self.gates = context.gates
        self.nets = context.nets
        self.bus_components = context.bus_components
        self.vmem_config = context.vmem_config
        self.virtual_display_config = context.virtual_display_config
        self.clock_interval = max(1, int(context.clock_interval))

        self.gate_count = max(self.gates, default=0)
        self.net_count = max(self.nets, default=0)
        self.bus_count = max(self.bus_components, default=0)

        self.ops: List[RuntimeOp] = []
        self.initial_work: List[int] = []
        self.random_op_ids: List[int] = []
        self.clock_ops: List[RuntimeOp] = []
        self.vmem_data_op_ids: List[int] = []
        self.readers_by_net: List[List[int]] = [[] for _ in range(self.net_count + 1)]
        self.bus_net_ids: Dict[int, Tuple[int, ...]] = {bus_id: tuple(sorted(component.net_ids)) for bus_id, component in self.bus_components.items()}

        self._build_plan()

        self.gate_out = bytearray(self.gate_count + 1)
        self.net_counts = [0 for _ in range(self.net_count + 1)]
        self.next_net_counts = [0 for _ in range(self.net_count + 1)]
        self.changed_net_flags = bytearray(self.net_count + 1)
        self.changed_net_ids: List[int] = []
        self.work: List[int] = []
        self.rng_state = [0 for _ in range(self.gate_count + 1)]

        self.clock_value = 0
        self.clock_ticks_until_toggle = 0
        self.tick_count = 0

        self.vmem_words = self._new_vmem_words()
        self.vmem_address_value = 0
        self.vmem_loaded_address = 0
        self.vmem_data_value = 0
        self.vmem_lock_pending = False
        self.vmem_deferred_publish = False

        self._next_work: List[int] = []
        self._next_work_flags = bytearray(len(self.ops))
        self._vmem_locked_this_tick = False
        self._vmem_address_changed_this_tick = False
        self._vmem_data_changed_this_tick = False

        # Cycle-accurate collection controls. When auto_clock/auto_random are
        # left True the runtime behaves exactly as before. The SMT observation
        # collector disables them so it can drive one settled clock period at a
        # time and evaluate RANDOM gates exactly once per cycle.
        self.auto_clock = True
        self.auto_random = True
        self._pending_clock: int | None = None
        self._pending_random = False
        self.vmem_dirty: set[int] = set()

        self.reset()

    @property
    def visual_blocks(self) -> Sequence[VisualBlock]:
        return self.ctx.visual_blocks

    @property
    def grid_width(self) -> int:
        return self.ctx.grid_width

    @property
    def grid_height(self) -> int:
        return self.ctx.grid_height

    def reset(self) -> None:
        self.gate_out = bytearray(self.gate_count + 1)
        self.net_counts = [0 for _ in range(self.net_count + 1)]
        self.next_net_counts = [0 for _ in range(self.net_count + 1)]
        self.changed_net_flags = bytearray(self.net_count + 1)
        self.changed_net_ids = []
        self.rng_state = [self._random_seed(gate_id) for gate_id in range(self.gate_count + 1)]
        for op in self.ops:
            op.input_mask = 0

        self.clock_value = 0
        self.clock_ticks_until_toggle = 0
        self.tick_count = 0

        self.vmem_words = self._new_vmem_words()
        self.vmem_address_value = 0
        self.vmem_loaded_address = 0
        self.vmem_data_value = self._vmem_word(0) & self._vmem_data_mask()
        self.vmem_lock_pending = False
        self.vmem_deferred_publish = False

        self._pending_clock = None
        self._pending_random = False
        self.vmem_dirty = set()

        for op in self.ops:
            if op.kind == "LATCH":
                value = 1 if op.initial_state else 0
                self.gate_out[op.gate_id] = value
                self._seed_outputs(op.output_net_ids, value)

        self._seed_vmem_outputs()
        self.work = list(self.initial_work)

    def tick(self, count: int = 1) -> None:
        for _ in range(max(0, int(count))):
            self._tick_once()

    def make_snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            tick=self.tick_count,
            gate_out=bytes(self.gate_out),
            net_state=self._net_state_bytes(),
            bus_state=self._bus_state_bytes(),
            vmem=self._vmem_snapshot(),
            vdisplay_words=self.copy_vdisplay_words(),
        )

    def copy_vdisplay_words(self) -> Tuple[int, ...]:
        display = self.virtual_display_config
        if not display.enabled or not self._vmem_enabled() or display.required_word_count <= 0:
            return ()
        pointer = self._vmem_word(display.pointer_address) & (self.vmem_config.WORD_COUNT - 1)
        mask = self.vmem_config.WORD_COUNT - 1
        return tuple(self._vmem_word((pointer + i) & mask) for i in range(display.required_word_count))

    def copy_vmem_words(self, start_address: int, word_count: int) -> Tuple[int, ...]:
        if not self._vmem_enabled():
            return ()
        start = max(0, int(start_address))
        end = min(self.vmem_config.WORD_COUNT, start + max(0, int(word_count)))
        return tuple(self._vmem_word(address) for address in range(start, end))

    def _build_plan(self) -> None:
        for gate_id in sorted(self.gates):
            gate = self.gates[gate_id]
            input_net_ids = tuple(net.id for net in gate.inputs)
            output_net_ids = tuple(net.id for net in sorted(gate.outputs, key=lambda net: net.id))

            if gate.kind == "CLOCK":
                self.clock_ops.append(
                    RuntimeOp(
                        gate_id=gate.id,
                        kind=gate.kind,
                        kind_code=0,
                        input_net_ids=input_net_ids,
                        output_net_ids=output_net_ids,
                    )
                )
                continue

            if gate.kind not in BASIC_KINDS | {
                "LED",
                "BREAKPOINT",
                "LATCH",
                "RANDOM",
                VMEM_ADDRESS_KIND,
                VMEM_DATA_KIND,
            }:
                continue

            op = RuntimeOp(
                gate_id=gate.id,
                kind=gate.kind,
                kind_code=KIND_CODES[gate.kind],
                input_net_ids=input_net_ids,
                output_net_ids=output_net_ids,
                initial_state=1 if gate.initial_state else 0,
                value_bit=gate.vmem_value_bit,
            )
            op_id = len(self.ops)
            self.ops.append(op)

            if op.kind == "RANDOM":
                self.random_op_ids.append(op_id)
            else:
                self.initial_work.append(op_id)
                for net_id in sorted(set(input_net_ids)):
                    if 0 < net_id <= self.net_count:
                        self.readers_by_net[net_id].append(op_id)

            if op.kind == VMEM_DATA_KIND:
                self.vmem_data_op_ids.append(op_id)

    def _tick_once(self) -> None:
        self.changed_net_ids.clear()
        self._next_work = []
        self._next_work_flags = bytearray(len(self.ops))
        self._vmem_locked_this_tick = self.vmem_lock_pending
        self._vmem_address_changed_this_tick = False
        self._vmem_data_changed_this_tick = False

        self._publish_deferred_vmem_outputs()
        self._step_clock_sources()

        current_work = self.work
        self.work = []
        if self.random_op_ids and (self.auto_random or self._pending_random):
            self._evaluate_op_ids(self.random_op_ids)
        self._pending_random = False
        if current_work:
            self._evaluate_op_ids(current_work)

        self._finalize_vmem_tick()
        self._publish_changed_nets()
        self.work = self._next_work
        self.tick_count += 1

    def _evaluate_op_ids(self, op_ids: Sequence[int]) -> None:
        ops = self.ops
        net_counts = self.net_counts
        gate_out = self.gate_out
        next_net_counts = self.next_net_counts
        changed_flags = self.changed_net_flags
        changed_ids = self.changed_net_ids
        rng_state = self.rng_state
        net_count = self.net_count
        vmem_locked = self._vmem_locked_this_tick

        for op_id in op_ids:
            op = ops[op_id]
            kind = op.kind_code
            input_net_ids = op.input_net_ids
            gate_id = op.gate_id

            if kind in OR_LIKE_CODES:
                value = 0
                for net_id in input_net_ids:
                    if 0 < net_id <= net_count and net_counts[net_id] != 0:
                        value = 1
                        break
            elif kind in NOT_LIKE_CODES:
                value = 1
                for net_id in input_net_ids:
                    if 0 < net_id <= net_count and net_counts[net_id] != 0:
                        value = 0
                        break
            elif kind == KIND_AND:
                value = 1 if input_net_ids else 0
                for net_id in input_net_ids:
                    if net_id <= 0 or net_id > net_count or net_counts[net_id] == 0:
                        value = 0
                        break
            elif kind == KIND_XOR:
                value = 0
                for net_id in input_net_ids:
                    if 0 < net_id <= net_count and net_counts[net_id] != 0:
                        value ^= 1
            elif kind == KIND_RANDOM:
                state = rng_state[gate_id]
                state ^= (state << 13) & 0xFFFFFFFF
                state ^= (state >> 17) & 0xFFFFFFFF
                state ^= (state << 5) & 0xFFFFFFFF
                state &= 0xFFFFFFFF
                rng_state[gate_id] = state
                enabled = True
                if input_net_ids:
                    enabled = False
                    for net_id in input_net_ids:
                        if 0 < net_id <= net_count and net_counts[net_id] != 0:
                            enabled = True
                            break
                value = (state & 1) if enabled else 0
            elif kind == KIND_LATCH:
                new_mask = 0
                for index, net_id in enumerate(input_net_ids):
                    if 0 < net_id <= net_count and net_counts[net_id] != 0:
                        new_mask |= 1 << index
                rising_mask = new_mask & ~op.input_mask
                op.input_mask = new_mask
                if not rising_mask:
                    continue
                value = 0 if gate_out[gate_id] else 1
            elif kind == KIND_VMEM_ADDRESS:
                new_mask = 0
                for index, net_id in enumerate(input_net_ids):
                    if 0 < net_id <= net_count and net_counts[net_id] != 0:
                        new_mask |= 1 << index
                rising_mask = new_mask & ~op.input_mask
                op.input_mask = new_mask
                if not (rising_mask.bit_count() & 1) or vmem_locked:
                    continue
                bit = op.value_bit or 0
                bit_mask = 1 << bit
                self.vmem_address_value = (self.vmem_address_value ^ bit_mask) & 0xFFFFFFFF
                self._vmem_address_changed_this_tick = True
                value = 1 if (self.vmem_address_value & bit_mask) else 0
            elif kind == KIND_VMEM_DATA:
                new_mask = 0
                for index, net_id in enumerate(input_net_ids):
                    if 0 < net_id <= net_count and net_counts[net_id] != 0:
                        new_mask |= 1 << index
                rising_mask = new_mask & ~op.input_mask
                op.input_mask = new_mask
                if not (rising_mask.bit_count() & 1) or vmem_locked:
                    continue
                self.vmem_data_value = (self.vmem_data_value ^ (1 << (op.value_bit or 0))) & 0xFFFFFFFF
                self._vmem_data_changed_this_tick = True
                continue
            else:
                continue

            old_value = gate_out[gate_id]
            if old_value == value:
                continue
            gate_out[gate_id] = value
            delta = value - old_value
            for net_id in op.output_net_ids:
                if not (0 < net_id <= net_count):
                    continue
                next_net_counts[net_id] += delta
                if changed_flags[net_id]:
                    continue
                changed_flags[net_id] = 1
                changed_ids.append(net_id)

    def _step_clock_sources(self) -> None:
        if not self.clock_ops:
            return

        if not self.auto_clock:
            if self._pending_clock is not None:
                self.clock_value = self._pending_clock & 1
                for op in self.clock_ops:
                    self._set_gate_and_commit(op, self.clock_value)
                self._pending_clock = None
            return

        if self.clock_ticks_until_toggle != 0:
            self.clock_ticks_until_toggle -= 1
            return

        self.clock_value ^= 1
        self.clock_ticks_until_toggle = self.clock_interval - 1
        for op in self.clock_ops:
            self._set_gate_and_commit(op, self.clock_value)

    def queue_clock(self, value: int) -> None:
        """Drive the clock to `value` on the next tick (manual-clock mode)."""
        self._pending_clock = int(value) & 1

    def queue_random(self) -> None:
        """Evaluate every RANDOM gate exactly once on the next tick."""
        self._pending_random = True

    def settle(self, max_ticks: int) -> bool:
        """Tick until the circuit is quiescent or `max_ticks` is reached.

        Returns True if the circuit settled (no scheduled work, no nets changed
        on the final tick, no deferred VMEM publish pending).
        """
        for _ in range(max(0, int(max_ticks))):
            self._tick_once()
            if not self.work and not self.changed_net_ids and not self.vmem_deferred_publish:
                return True
        return False

    def _publish_deferred_vmem_outputs(self) -> None:
        if not self.vmem_deferred_publish:
            return
        for op_id in self.vmem_data_op_ids:
            op = self.ops[op_id]
            self._set_gate_and_commit(op, self._bit_value(self.vmem_data_value, op.value_bit))
        self.vmem_deferred_publish = False

    def _finalize_vmem_tick(self) -> None:
        if not self._vmem_enabled():
            return
        if self._vmem_data_changed_this_tick:
            current_word = self._vmem_word(self.vmem_loaded_address)
            preserved = current_word & (~self._vmem_data_mask() & 0xFFFFFFFF)
            self.vmem_words[self.vmem_loaded_address] = (preserved | (self.vmem_data_value & self._vmem_data_mask())) & 0xFFFFFFFF
            self.vmem_dirty.add(self.vmem_loaded_address)
            self.vmem_deferred_publish = True

        if self._vmem_locked_this_tick:
            self.vmem_lock_pending = False
        elif self._vmem_address_changed_this_tick:
            self.vmem_loaded_address = self.vmem_address_value & self._vmem_address_mask()
            self.vmem_data_value = self._vmem_word(self.vmem_loaded_address) & self._vmem_data_mask()
            self.vmem_deferred_publish = True
            self.vmem_lock_pending = True

    def _publish_changed_nets(self) -> None:
        net_counts = self.net_counts
        next_net_counts = self.next_net_counts
        changed_flags = self.changed_net_flags
        readers_by_net = self.readers_by_net
        next_work_flags = self._next_work_flags
        next_work = self._next_work
        for net_id in self.changed_net_ids:
            old_on = net_counts[net_id] != 0
            new_on = next_net_counts[net_id] != 0
            net_counts[net_id] = next_net_counts[net_id]
            changed_flags[net_id] = 0
            if old_on == new_on:
                continue
            for op_id in readers_by_net[net_id]:
                if next_work_flags[op_id]:
                    continue
                next_work_flags[op_id] = 1
                next_work.append(op_id)

    def _set_gate_and_commit(self, op: RuntimeOp, value: int) -> None:
        old_value = self.gate_out[op.gate_id]
        if old_value == value:
            return
        self.gate_out[op.gate_id] = value
        self._commit_outputs(op.output_net_ids, old_value, value)

    def _seed_outputs(self, output_net_ids: Iterable[int], value: int) -> None:
        if not value:
            return
        for net_id in output_net_ids:
            if 0 < net_id <= self.net_count:
                self.net_counts[net_id] += 1
                self.next_net_counts[net_id] += 1

    def _commit_outputs(self, output_net_ids: Iterable[int], old_value: int, value: int) -> None:
        delta = value - old_value
        if delta == 0:
            return
        next_net_counts = self.next_net_counts
        changed_flags = self.changed_net_flags
        changed_ids = self.changed_net_ids
        net_count = self.net_count
        for net_id in output_net_ids:
            if not (0 < net_id <= net_count):
                continue
            next_net_counts[net_id] += delta
            if changed_flags[net_id]:
                continue
            changed_flags[net_id] = 1
            changed_ids.append(net_id)

    def _seed_vmem_outputs(self) -> None:
        if not self._vmem_enabled():
            return
        for op in self.ops:
            if op.kind == VMEM_ADDRESS_KIND:
                value = self._bit_value(self.vmem_address_value, op.value_bit)
            elif op.kind == VMEM_DATA_KIND:
                value = self._bit_value(self.vmem_data_value, op.value_bit)
            else:
                continue
            self.gate_out[op.gate_id] = value
            self._seed_outputs(op.output_net_ids, value)

    def _net_state_bytes(self) -> bytes:
        state = bytearray(self.net_count + 1)
        for net_id in range(1, self.net_count + 1):
            state[net_id] = 1 if self.net_counts[net_id] else 0
        return bytes(state)

    def _bus_state_bytes(self) -> bytes:
        state = bytearray(self.bus_count + 1)
        for bus_id, net_ids in self.bus_net_ids.items():
            if bus_id <= self.bus_count and any(0 < net_id <= self.net_count and self.net_counts[net_id] for net_id in net_ids):
                state[bus_id] = 1
        return bytes(state)

    def _vmem_snapshot(self) -> VMemSnapshot:
        enabled = self._vmem_enabled()
        return VMemSnapshot(
            enabled=enabled,
            word_count=self.vmem_config.WORD_COUNT if enabled else 0,
            address_bits=self.vmem_config.address_bits if enabled else 0,
            address_mask=self._vmem_address_mask() if enabled else 0,
            data_bits=self.vmem_config.data_bits if enabled else 0,
            data_mask=self._vmem_data_mask() if enabled else 0,
            address_value=self.vmem_address_value,
            active_address=self.vmem_loaded_address,
            data_value=self.vmem_data_value,
            lock_pending=self.vmem_lock_pending,
        )

    def _new_vmem_words(self) -> array:
        word_count = self.vmem_config.WORD_COUNT if self._vmem_enabled() else 1
        if array("I").itemsize != 4:
            raise RuntimeError("runner VMEM storage requires 32-bit array('I') items.")
        raw = self.vmem_config.initial_data_be if self._vmem_enabled() else b""
        if raw:
            words = array("I")
            words.frombytes(raw)
            if sys.byteorder == "little":
                words.byteswap()
            if len(words) == word_count:
                return words
        return array("I", [0]) * word_count

    def _vmem_word(self, address: int) -> int:
        if not self._vmem_enabled():
            return 0
        return int(self.vmem_words[address & (self.vmem_config.WORD_COUNT - 1)])

    def _vmem_enabled(self) -> bool:
        return self.vmem_config is not None and self.vmem_config.enabled

    def _vmem_address_mask(self) -> int:
        return self.vmem_config.address_mask if self._vmem_enabled() else 0

    def _vmem_data_mask(self) -> int:
        return self.vmem_config.data_mask if self._vmem_enabled() else 0

    @staticmethod
    def _bit_value(value: int, bit_index: int | None) -> int:
        return 0 if bit_index is None else ((value >> bit_index) & 1)

    @staticmethod
    def _random_seed(gate_id: int) -> int:
        seed = (0x9E3779B9 ^ gate_id) & 0xFFFFFFFF
        return seed or 1


class SimulationDriver:
    def __init__(
        self,
        runtime: CircuitRuntime,
        *,
        ticks_per_second: int | None = None,
        frame_hz: int = 60,
        max_burst_ticks: int = 1024,
    ):
        self.runtime = runtime
        self.ticks_per_second = ticks_per_second
        self.frame_hz = max(1, int(frame_hz))
        self.max_burst_ticks = max(1, int(max_burst_ticks))
        self.paused = False
        self.max_mode = ticks_per_second is None

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot_lock = threading.Lock()
        self._snapshot = self.runtime.make_snapshot()
        self._reset_requested = False
        self._step_requests = 0
        self._last_tps = 0.0

    @property
    def last_tps(self) -> float:
        return self._last_tps

    def snapshot(self) -> RuntimeSnapshot:
        with self._snapshot_lock:
            return self._snapshot

    def start(self) -> None:
        if self._thread is not None:
            return
        sys.setswitchinterval(0.001)
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="vcb-python-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def request_reset(self) -> None:
        self._reset_requested = True

    def request_step(self, count: int = 1) -> None:
        self._step_requests += max(1, int(count))

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def toggle_max_mode(self) -> None:
        self.max_mode = not self.max_mode

    def set_ticks_per_second(self, value: int | None) -> None:
        self.ticks_per_second = None if value is None else max(0, int(value))
        self.max_mode = value is None

    def _run(self) -> None:
        next_frame = time.perf_counter()
        tick_accumulator = 0
        tick_counter_start = self.runtime.tick_count
        tps_window_start = time.perf_counter()

        while self._running.is_set():
            frame_start = time.perf_counter()
            self._consume_reset_if_needed()

            if self.paused:
                steps = self._step_requests
                self._step_requests = 0
                if steps:
                    self.runtime.tick(steps)
            elif self.max_mode:
                deadline = frame_start + (1.0 / self.frame_hz)
                while self._running.is_set() and time.perf_counter() < deadline:
                    self._consume_reset_if_needed()
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    tick_for = getattr(self.runtime, "tick_for", None)
                    if tick_for is not None:
                        tick_for(remaining)
                    else:
                        self.runtime.tick(self.max_burst_ticks)
                    if self.paused or not self.max_mode:
                        break
            else:
                tps = max(0, int(self.ticks_per_second or 0))
                ticks = tps // self.frame_hz
                tick_accumulator += tps % self.frame_hz
                if tick_accumulator >= self.frame_hz:
                    ticks += tick_accumulator // self.frame_hz
                    tick_accumulator %= self.frame_hz
                if ticks:
                    self.runtime.tick(ticks)

            self._publish_snapshot()

            now = time.perf_counter()
            if now - tps_window_start >= 0.5:
                tick_delta = max(0, self.runtime.tick_count - tick_counter_start)
                self._last_tps = tick_delta / max(0.000001, now - tps_window_start)
                tick_counter_start = self.runtime.tick_count
                tps_window_start = now

            next_frame += 1.0 / self.frame_hz
            sleep_for = next_frame - time.perf_counter()
            if sleep_for > 0 and not (self.max_mode and not self.paused):
                time.sleep(min(sleep_for, 0.01))
            elif sleep_for < -1.0:
                next_frame = time.perf_counter()

    def _consume_reset_if_needed(self) -> None:
        if not self._reset_requested:
            return
        self.runtime.reset()
        self._reset_requested = False
        self._publish_snapshot()

    def _publish_snapshot(self) -> None:
        snapshot = self.runtime.make_snapshot()
        with self._snapshot_lock:
            self._snapshot = snapshot
