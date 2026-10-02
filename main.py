"""Compile a VCB project and run the native event simulator at full speed."""
import argparse
import struct
import time
from pathlib import Path

from compiler.assembler import assemble_project_assembly_words, project_uses_assembly
from compiler.compiler import BlockConfig, CircuitCompiler, VirtualDisplayConfig, VMemConfig, VMemLayout
from compiler.extractor import extract_layer_from_project, extract_vmem_from_project, load_project
from runner.context import EmitterContext
from settings import LOGIC_LAYER_INDEX, VCB_ROW_WIDTH

BLOCKS = {
    # MISC
    "BREAKPOINT": 0xFF0000E0,
    "LATCH_ON": 0xFF9FFF63,
    "LATCH_OFF": 0xFF474D38,
    "CLOCK": 0xFF4100FF,
    "RANDOM": 0xFF00FFE5,
    "CROSS": 0xFF8E7866,
    "BUFFER": 0xFF63FF92,
    "TUNNEL": 0xFF725553,
    "LED": 0xFFFFFFFF,
    "FILLER": 0xFFA1AB8C,
    # R/W
    "WRITE": 0xFF3E384D,
    "READ": 0xFF5D472E,
    # GATES
    "NOR": 0xFFFFD930,
    "OR": 0xFFFFF263,
    "NOT": 0xFF8A62FF,
    "AND": 0xFF63C6FF,
    "XOR": 0xFFFF74AE,
    # WIRES
    "WIRE_0": 0xFF9755A1,
    "WIRE_1": 0xFFA15687,
    "WIRE_2": 0xFFA15666,
    "WIRE_3": 0xFFA16256,
    "WIRE_4": 0xFFA17B56,
    "WIRE_5": 0xFFA19356,
    "WIRE_6": 0xFF8DA156,
    "WIRE_7": 0xFF56A16C,
    "WIRE_8": 0xFF56A188,
    "WIRE_9": 0xFF56A199,
    "WIRE_10": 0xFF5698A1,
    "WIRE_11": 0xFF5685A1,
    "WIRE_12": 0xFF566CA1,
    "WIRE_13": 0xFF5E55A1,
    "WIRE_14": 0xFFAEA89F,
    "WIRE_15": 0xFF41352A,
    # BUSSES
    "BUS_0": 0xFF24707A,
    "BUS_1": 0xFF662D7A,
    "BUS_2": 0xFF7A6225,
    "BUS_3": 0xFF7A4124,
    "BUS_4": 0xFF247A3E,
    "BUS_5": 0xFF242F7A,
}


def rgb(r: int, g: int, b: int) -> int:
    """
    Pack RGB bytes into the little-endian uint32 format used by VCB logic layers.
    Alpha is always forced to fully opaque.

    Example:
      rgb(255, 0, 65) -> 0xFF4100FF
    """
    return (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | 0xFF000000


# Optional renderer-only palette overrides keyed by block name.
# These do not affect block recognition or simulation IDs.
# Values are written in normal RGB channel order for readability.
VISUAL_COLOR_OVERRIDES = {
    # "CLOCK": rgb(255, 0, 65),
    # "AND": rgb(255, 198, 99),
}




def build_config() -> BlockConfig:
    gate_kinds = {
        BLOCKS["BUFFER"]: "BUFFER",
        BLOCKS["LED"]: "LED",
        BLOCKS["NOR"]: "NOR",
        BLOCKS["OR"]: "OR",
        BLOCKS["NOT"]: "NOT",
        BLOCKS["AND"]: "AND",
        BLOCKS["XOR"]: "XOR",
        BLOCKS["LATCH_ON"]: "LATCH",
        BLOCKS["LATCH_OFF"]: "LATCH",
        BLOCKS["CLOCK"]: "CLOCK",
        BLOCKS["RANDOM"]: "RANDOM",
        BLOCKS["BREAKPOINT"]: "BREAKPOINT",
    }

    gate_initial_states = {
        BLOCKS["LATCH_ON"]: 1,
        BLOCKS["LATCH_OFF"]: 0,
    }

    return BlockConfig(
        empty_ids={0},
        wire_ids={BLOCKS[f"WIRE_{i}"] for i in range(16)},
        bus_ids={BLOCKS[f"BUS_{i}"] for i in range(6)},
        crossroad_ids={BLOCKS["CROSS"]},
        tunnel_ids={BLOCKS["TUNNEL"]},
        gate_ids=set(gate_kinds),
        gate_kinds=gate_kinds,
        gate_initial_states=gate_initial_states,
        read_ids={BLOCKS["READ"]},
        write_ids={BLOCKS["WRITE"]},
    )


def build_visual_color_palette() -> dict[int, int]:
    palette = dict(BLOCKS)
    for name, color in VISUAL_COLOR_OVERRIDES.items():
        block_id = BLOCKS.get(name)
        if block_id is None:
            raise KeyError(f"Unknown visual override block name: {name}")
        palette[block_id] = color
    return palette


def assign_visual_colors(visual_blocks):
    palette = build_visual_color_palette()
    for block in visual_blocks:
        if block.base_color is not None:
            continue
        block.base_color = palette.get(block.block_id, block.block_id)
    return visual_blocks


def load_grid_from_project_data(project_data: dict, layer_index: int = LOGIC_LAYER_INDEX, row_width: int = VCB_ROW_WIDTH) -> list[list[int]]:
    raw_layer_bytes = extract_layer_from_project(project_data, layer_index)
    num_blocks = len(raw_layer_bytes) // 4
    flat_data = struct.unpack(f"<{num_blocks}I", raw_layer_bytes[: num_blocks * 4])
    return [list(flat_data[i : i + row_width]) for i in range(0, len(flat_data), row_width)]


def load_grid(project_path: str, layer_index: int, row_width: int) -> list[list[int]]:
    project_data = load_project(project_path)
    return load_grid_from_project_data(project_data, layer_index, row_width)


def _overlay_vmem_image(base_image: bytearray, overlay: bytes, source_name: str) -> None:
    expected_size = VMemConfig.WORD_COUNT * 4
    if not overlay:
        return
    if len(overlay) != expected_size:
        raise ValueError(f"{source_name} VMEM payload must be exactly {expected_size} bytes, got {len(overlay)}.")
    base_image[:] = overlay


def build_initial_vmem_image(project_data: dict, project_path: str | Path) -> bytes:
    image = bytearray(VMemConfig.WORD_COUNT * 4)

    stored_vmem = extract_vmem_from_project(project_data)
    _overlay_vmem_image(image, stored_vmem, "Stored project")

    if project_uses_assembly(project_data):
        assembly_words = assemble_project_assembly_words(project_data, project_path)
        for address, value in assembly_words.items():
            offset = address * 4
            image[offset : offset + 4] = value.to_bytes(4, byteorder="big", signed=False)

    return bytes(image)


def build_vmem_config(project_data: dict, project_path: str | Path) -> VMemConfig | None:
    uses_assembly = project_uses_assembly(project_data)
    if uses_assembly and not project_data.get("is_vmem_enabled"):
        raise ValueError("Assembly projects require VMem to be enabled.")

    if not project_data.get("is_vmem_enabled"):
        return None

    settings = project_data.get("vmem_settings")
    if not isinstance(settings, list) or len(settings) < 16:
        raise ValueError("Enabled VMem projects must contain a 16-entry 'vmem_settings' array.")

    address_layout = VMemLayout(*[int(value) for value in settings[0:7]])
    data_layout = VMemLayout(*[int(value) for value in settings[7:14]])
    initial_data_be = build_initial_vmem_image(project_data, project_path)
    return VMemConfig(
        enabled=True,
        address_layout=address_layout,
        data_layout=data_layout,
        persistent_start=int(settings[14]),
        persistent_end=int(settings[15]),
        initial_data_be=initial_data_be,
    )


def parse_rgb_hex(value: str | int) -> int:
    if isinstance(value, int):
        return value & 0xFFFFFF

    normalized = str(value).strip().lower()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    if len(normalized) != 6:
        raise ValueError(f"Expected a 6-digit RGB color, got {value!r}.")
    return int(normalized, 16) & 0xFFFFFF


def build_virtual_display_config(
    project_data: dict,
    vmem_config: VMemConfig | None,
) -> VirtualDisplayConfig | None:
    if not project_data.get("vdisplay_is_enabled"):
        return None

    settings = project_data.get("vdisplay_settings")
    if not isinstance(settings, list) or len(settings) < 8:
        raise ValueError("Enabled Virtual Display projects must contain an 8-entry 'vdisplay_settings' array.")

    palette_values = project_data.get("vdisplay_palette") or []
    if not isinstance(palette_values, list):
        raise ValueError("Virtual Display palette must be stored as a list.")

    color_depth = int(project_data.get("vdisplay_color_depth", 1))
    return VirtualDisplayConfig(
        enabled=True,
        position_x=int(settings[0]),
        position_y=int(settings[1]),
        resolution_x=int(settings[2]),
        resolution_y=int(settings[3]),
        scale_x=int(settings[4]),
        scale_y=int(settings[5]),
        pointer_address=int(settings[6]),
        word_size=int(settings[7]),
        color_depth=color_depth,
        direction=int(project_data.get("vdisplay_direction", 0)),
        palette_rgb=tuple(parse_rgb_hex(value) for value in palette_values),
    )


def build_clock_interval(project_data: dict) -> int:
    raw_interval = project_data.get("clock_interval", 1)
    try:
        interval = int(raw_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"clock_interval must be an integer, got {raw_interval!r}.") from exc
    if interval < 1:
        raise ValueError(f"clock_interval must be at least 1, got {interval}.")
    return interval


def compile_project(project_path: str):
    project_data = load_project(project_path)
    grid = load_grid_from_project_data(project_data)
    vmem_config = build_vmem_config(project_data, project_path)
    virtual_display_config = build_virtual_display_config(project_data, vmem_config)
    clock_interval = build_clock_interval(project_data)
    compiler = CircuitCompiler(
        grid,
        build_config(),
        vmem_config=vmem_config,
        virtual_display_config=virtual_display_config,
        clock_interval=clock_interval,
    )
    gates, nets = compiler.compile()
    regions = compiler.build_regions()
    return compiler, gates, nets, regions


def build_runner_context(project_path: str, *, print_info: bool = False) -> EmitterContext:
    compiler, gates, nets, regions = compile_project(project_path)
    if print_info:
        print(f"gates={len(gates)} nets={len(nets)} engine=native-event", flush=True)
    return EmitterContext.create(
        gates, nets, regions, assign_visual_colors(compiler.build_visual_blocks()),
        compiler.width, compiler.height, bus_components=compiler.bus_components,
        vmem_config=compiler.vmem_config, virtual_display_config=compiler.virtual_display_config,
        clock_interval=compiler.clock_interval,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Example: python main.py --project "path/to/project.vcb"',
    )
    parser.add_argument("--project", type=Path, required=True, metavar="FILE.vcb", help="VCB project to run")
    parser.add_argument("--compiler", default="gcc", help="C compiler (default: gcc)")
    parser.add_argument("--benchmark", type=float, metavar="SECONDS", help="Run without a window and report ticks per second")
    args = parser.parse_args(argv)
    if args.benchmark is not None and (not 0 < args.benchmark < float("inf")):
        parser.error("--benchmark must be a positive finite number")
    return args


def main():
    args = parse_args()
    if not args.project.is_file():
        raise SystemExit(f"Project not found: {args.project}")
    from runner.native import NativeRuntime
    runtime = NativeRuntime(build_runner_context(str(args.project), print_info=True), compiler=args.compiler)
    if args.benchmark is not None:
        start = time.perf_counter()
        while True:
            remaining = args.benchmark - (time.perf_counter() - start)
            if remaining <= 0:
                break
            runtime.tick_for(min(1.0 / 30.0, remaining))
        elapsed = time.perf_counter() - start
        print(f"ticks={runtime.tick_count} seconds={elapsed:.3f} tps={runtime.tick_count / elapsed:,.0f}")
        return
    from runner.app import run_prepared_runtime
    run_prepared_runtime(runtime)


if __name__ == "__main__":
    main()
