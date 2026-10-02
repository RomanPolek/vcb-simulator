"""Native event engine and immutable renderer snapshots.

ctypes releases the GIL during tick batches, allowing rendering in parallel.
"""
import ctypes
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from runner.simulation import CircuitRuntime, RuntimeSnapshot, VMemSnapshot

def _array(ctype: str, name: str, values: list[int]) -> str:
    size = max(1, len(values))
    if not values:
        values = [0]
    lines = [f"static const {ctype} {name}[{size}] = {{"]
    for start in range(0, len(values), 12):
        lines.append("    " + ", ".join(str(value) for value in values[start:start + 12]) + ",")
    lines.append("};")
    return "\n".join(lines)


def render_plan(runtime: CircuitRuntime) -> str:
    """Emit a netlist-specific table; preserve Python op and fanout order."""
    inputs: list[int] = []
    outputs: list[int] = []

    def packed(op) -> str:
        input_start, output_start = len(inputs), len(outputs)
        inputs.extend(op.input_net_ids)
        outputs.extend(op.output_net_ids)
        bit = 255 if op.value_bit is None else op.value_bit
        if op.kind in {"VMEM_ADDRESS", "VMEM_DATA"} and bit != 255 and not 0 <= bit < 32:
            raise ValueError(f"VMEM gate {op.gate_id} has invalid bit {bit}")
        return ("    {" + ", ".join(str(value) for value in (
            op.gate_id, input_start, len(op.input_net_ids), output_start,
            len(op.output_net_ids), op.kind_code, bit, op.initial_state)) + "},")

    ops = [packed(op) for op in runtime.ops]
    clocks = [packed(op) for op in runtime.clock_ops]
    readers: list[int] = []
    offsets = [0] * (runtime.net_count + 2)
    for net in range(1, runtime.net_count + 1):
        offsets[net] = len(readers)
        readers.extend(runtime.readers_by_net[net])
    offsets[runtime.net_count + 1] = len(readers)
    vmem = runtime.vmem_config
    enabled = runtime._vmem_enabled()
    word_count = vmem.WORD_COUNT if enabled else 1
    constants = {
        "NATIVE_GATE_COUNT": runtime.gate_count,
        "NATIVE_NET_COUNT": runtime.net_count,
        "NATIVE_OP_COUNT": len(runtime.ops),
        "NATIVE_INPUT_COUNT": len(inputs),
        "NATIVE_CLOCK_COUNT": len(clocks),
        "NATIVE_CLOCK_INTERVAL": runtime.clock_interval,
        "NATIVE_RANDOM_COUNT": len(runtime.random_op_ids),
        "NATIVE_INITIAL_WORK_COUNT": len(runtime.initial_work),
        "NATIVE_VMEM_DATA_COUNT": len(runtime.vmem_data_op_ids),
        "NATIVE_VMEM_ENABLED": int(enabled),
        "NATIVE_VMEM_COUNT": word_count,
        "NATIVE_VMEM_ADDRESS_MASK": runtime._vmem_address_mask(),
        "NATIVE_VMEM_DATA_MASK": runtime._vmem_data_mask(),
    }
    lines = [f"#define {key} {value}u" for key, value in constants.items()]
    lines.extend((
        _array("uint32_t", "native_inputs", inputs),
        _array("uint32_t", "native_outputs", outputs),
        "static const NativeOp native_ops[NATIVE_OP_COUNT + 1] = {\n" +
        "\n".join(ops or ["    {0},"]) + "\n};",
        "static const NativeOp native_clocks[NATIVE_CLOCK_COUNT + 1] = {\n" +
        "\n".join(clocks or ["    {0},"]) + "\n};",
        _array("uint32_t", "native_reader_offsets", offsets),
        _array("uint32_t", "native_reader_ops", readers),
        _array("uint32_t", "native_random_ops", runtime.random_op_ids),
        _array("uint32_t", "native_initial_work", runtime.initial_work),
        _array("uint32_t", "native_vmem_data_ops", runtime.vmem_data_op_ids),
    ))
    return "\n\n".join(lines) + "\n"



class NativeRuntime:
    def __init__(self, context, *, compiler="gcc"):
        self.ctx = context
        self.visual_blocks = context.visual_blocks
        self.grid_width, self.grid_height = context.grid_width, context.grid_height
        self.virtual_display_config = context.virtual_display_config
        self.vmem_config = context.vmem_config
        self.gate_count, self.net_count, self.bus_count = context.gate_count, context.net_count, context.bus_count
        plan = CircuitRuntime(context)
        self._initial_memory = plan.vmem_words
        self._bus_nets = plan.bus_net_ids
        source = Path(__file__).with_suffix(".c")
        header = render_plan(plan)
        compiler_path = shutil.which(compiler)
        if compiler_path is None:
            raise RuntimeError(f"C compiler not found: {compiler}. Install GCC or pass --compiler with its path.")
        fingerprint = hashlib.sha256(source.read_bytes() + header.encode() + compiler_path.encode()).hexdigest()[:20]
        destination = source.parent.parent / "output" / fingerprint
        destination.mkdir(parents=True, exist_ok=True)
        library = destination / ("event.dll" if sys.platform == "win32" else "event.so")
        if not library.exists():
            (destination / "native_plan.h").write_text(header, encoding="utf-8")
            command = [compiler_path, "-O3", "-march=native", "-std=c11", "-shared", "-I", str(destination), str(source), "-o", str(library)]
            if sys.platform != "win32":
                command.append("-fPIC")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"Native engine build failed:\n{result.stderr or result.stdout}")
        # Each runtime needs its own C globals, including when tests use the same netlist.
        import tempfile
        self._library_dir = tempfile.TemporaryDirectory(prefix="vcb-event-")
        instance = Path(self._library_dir.name) / library.name
        shutil.copy2(library, instance)
        self.native = ctypes.CDLL(str(instance))
        import weakref
        if sys.platform == "win32":
            from _ctypes import FreeLibrary as unload
        else:
            from _ctypes import dlclose as unload
        def release(handle, directory):
            unload(handle)
            directory.cleanup()
        self._finalizer = weakref.finalize(self, release, self.native._handle, self._library_dir)
        n = self.native
        for name in ("native_tick_many", "native_tick_for", "native_load_memory", "native_reset_state"):
            getattr(n, name).restype = None
        n.native_tick_many.argtypes = [ctypes.c_uint64]
        n.native_tick_for.argtypes = [ctypes.c_double]
        n.native_load_memory.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
        n.native_tick_count.restype = ctypes.c_uint64
        for name in ("native_vmem_address", "native_vmem_loaded", "native_vmem_data", "native_vmem_lock"):
            getattr(n, name).restype = ctypes.c_uint32
        for name, ctype in (("native_gate_out", ctypes.c_uint8), ("native_net_counts", ctypes.c_int32), ("native_memory", ctypes.c_uint32)):
            getattr(n, name).restype = ctypes.POINTER(ctype)
        self._gates = np.ctypeslib.as_array(n.native_gate_out(), shape=(self.gate_count + 1,))
        self._nets = np.ctypeslib.as_array(n.native_net_counts(), shape=(self.net_count + 1,))
        self._memory = np.ctypeslib.as_array(n.native_memory(), shape=(len(self._initial_memory),))
        self.reset()

    @property
    def tick_count(self):
        return self.native.native_tick_count()

    def tick(self, count=1):
        self.native.native_tick_many(max(0, int(count)))

    def tick_for(self, seconds):
        """Run native ticks for a wall-time budget, releasing the GIL throughout."""
        self.native.native_tick_for(seconds)

    def reset(self):
        words = self._initial_memory
        self.native.native_load_memory((ctypes.c_uint32 * len(words)).from_buffer(words), len(words))
        self.native.native_reset_state()

    def make_snapshot(self):
        n = self.native
        nets = (self._nets != 0).astype(np.uint8)
        buses = bytearray(self.bus_count + 1)
        for bus, ids in self._bus_nets.items():
            buses[bus] = any(nets[i] for i in ids if 0 < i <= self.net_count)
        config = self.vmem_config
        vmem = VMemSnapshot(config.enabled, config.WORD_COUNT if config.enabled else 0,
            config.address_bits, config.address_mask, config.data_bits, config.data_mask,
            n.native_vmem_address(), n.native_vmem_loaded(), n.native_vmem_data(), bool(n.native_vmem_lock()))
        display = self.virtual_display_config
        words = ()
        if display.enabled and config.enabled and display.required_word_count > 0:
            mask = config.WORD_COUNT - 1
            pointer = int(self._memory[display.pointer_address & mask]) & mask
            indices = (np.arange(display.required_word_count) + pointer) & mask
            words = tuple(self._memory[indices].tolist())
        return RuntimeSnapshot(self.tick_count, self._gates.tobytes(), nets.tobytes(), bytes(buses), vmem, words)
