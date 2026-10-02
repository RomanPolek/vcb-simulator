import collections
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


TunnelSignature = Tuple[str, Union[int, str]]
VMEM_ADDRESS_KIND = "VMEM_ADDRESS"
VMEM_DATA_KIND = "VMEM_DATA"


@dataclass
class BlockConfig:
    """Define which integer IDs correspond to which mechanics."""

    empty_ids: Set[int] = field(default_factory=lambda: {0})
    wire_ids: Set[int] = field(default_factory=set)
    bus_ids: Set[int] = field(default_factory=set)
    crossroad_ids: Set[int] = field(default_factory=set)
    tunnel_ids: Set[int] = field(default_factory=set)
    gate_ids: Set[int] = field(default_factory=set)
    gate_kinds: Dict[int, str] = field(default_factory=dict)
    gate_initial_states: Dict[int, int] = field(default_factory=dict)
    read_ids: Set[int] = field(default_factory=set)
    write_ids: Set[int] = field(default_factory=set)

    def is_trace_wire(self, block_id: int) -> bool:
        """Read and Write blocks are treated like wires for logical net tracing."""
        return block_id in self.wire_ids or block_id in self.read_ids or block_id in self.write_ids

    def is_bus(self, block_id: int) -> bool:
        return block_id in self.bus_ids

    def is_any_wire(self, block_id: int) -> bool:
        return self.is_trace_wire(block_id) or self.is_bus(block_id)

    def is_gate(self, block_id: int) -> bool:
        return block_id in self.gate_ids

    def get_gate_kind(self, block_id: int) -> Optional[str]:
        if not self.is_gate(block_id):
            return None
        return self.gate_kinds.get(block_id, f"GATE_{block_id:08X}")


@dataclass(frozen=True)
class VMemLayout:
    bits: int
    position_x: int
    position_y: int
    offset_x: int
    offset_y: int
    size_x: int
    size_y: int

    def bit_origin(self, bit_index: int) -> Tuple[int, int]:
        return (
            self.position_x + (self.offset_x * bit_index),
            self.position_y + (self.offset_y * bit_index),
        )

    def bit_blocks(self, bit_index: int) -> Set[Tuple[int, int]]:
        origin_x, origin_y = self.bit_origin(bit_index)
        return {(origin_x + offset_x, origin_y + offset_y) for offset_y in range(self.size_y) for offset_x in range(self.size_x)}


@dataclass(frozen=True)
class VMemConfig:
    WORD_COUNT: int = 1 << 20
    MAX_ADDRESS_BITS: int = 20
    MAX_DATA_BITS: int = 32

    enabled: bool = False
    address_layout: Optional[VMemLayout] = None
    data_layout: Optional[VMemLayout] = None
    persistent_start: int = 0
    persistent_end: int = 0
    initial_data_be: bytes = b""

    @property
    def address_bits(self) -> int:
        return 0 if self.address_layout is None else self.address_layout.bits

    @property
    def data_bits(self) -> int:
        return 0 if self.data_layout is None else self.data_layout.bits

    @property
    def address_mask(self) -> int:
        if self.address_bits <= 0:
            return 0
        return (1 << self.address_bits) - 1

    @property
    def data_mask(self) -> int:
        if self.data_bits <= 0:
            return 0
        if self.data_bits >= self.MAX_DATA_BITS:
            return 0xFFFFFFFF
        return (1 << self.data_bits) - 1

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.address_layout is None or self.data_layout is None:
            raise ValueError("Enabled VMem config requires both address and data layouts.")
        if not (0 < self.address_bits <= self.MAX_ADDRESS_BITS):
            raise ValueError(f"VMem address bits must be between 1 and {self.MAX_ADDRESS_BITS}.")
        if not (0 < self.data_bits <= self.MAX_DATA_BITS):
            raise ValueError(f"VMem data bits must be between 1 and {self.MAX_DATA_BITS}.")
        if self.initial_data_be and len(self.initial_data_be) != self.WORD_COUNT * 4:
            raise ValueError(f"VMem payload must be exactly {self.WORD_COUNT * 4} bytes, got {len(self.initial_data_be)}.")


@dataclass(frozen=True)
class VirtualDisplayConfig:
    INDEXED_MAX_COLOR_DEPTH: int = 8
    RGB_COLOR_DEPTH: int = 24
    MAX_PIXEL_COUNT: int = 1 << 18

    enabled: bool = False
    position_x: int = 0
    position_y: int = 0
    resolution_x: int = 0
    resolution_y: int = 0
    scale_x: int = 1
    scale_y: int = 1
    pointer_address: int = 0
    word_size: int = 32
    color_depth: int = 1
    direction: int = 0
    palette_rgb: Tuple[int, ...] = ()

    @property
    def is_rgb_mode(self) -> bool:
        return self.color_depth == self.RGB_COLOR_DEPTH

    @property
    def palette_entry_count(self) -> int:
        if self.is_rgb_mode:
            return 0
        return 1 << self.color_depth

    @property
    def pixel_count(self) -> int:
        return self.resolution_x * self.resolution_y

    @property
    def pixels_per_word(self) -> int:
        bits_per_pixel = self.RGB_COLOR_DEPTH if self.is_rgb_mode else self.color_depth
        if bits_per_pixel <= 0:
            return 0
        return self.word_size // bits_per_pixel

    @property
    def required_word_count(self) -> int:
        pixels_per_word = self.pixels_per_word
        if pixels_per_word <= 0:
            return 0
        return (self.pixel_count + pixels_per_word - 1) // pixels_per_word

    def validate(self, vmem_config: Optional[VMemConfig]) -> None:
        if not self.enabled:
            return
        if vmem_config is None or not vmem_config.enabled:
            raise ValueError("Virtual Display requires VMem to be enabled.")
        if self.resolution_x <= 0 or self.resolution_y <= 0:
            raise ValueError("Virtual Display resolution must be positive on both axes.")
        if self.pixel_count > self.MAX_PIXEL_COUNT:
            raise ValueError(f"Virtual Display supports at most {self.MAX_PIXEL_COUNT} pixels, got {self.pixel_count}.")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("Virtual Display scale must be positive on both axes.")
        if not (0 <= self.pointer_address < VMemConfig.WORD_COUNT):
            raise ValueError(f"Virtual Display pointer address must be between 0 and {VMemConfig.WORD_COUNT - 1}.")
        if not (1 <= self.word_size <= VMemConfig.MAX_DATA_BITS):
            raise ValueError(f"Virtual Display word size must be between 1 and {VMemConfig.MAX_DATA_BITS}.")
        if self.direction not in {0, 1}:
            raise ValueError("Virtual Display direction must be 0 (horizontal) or 1 (vertical).")
        if self.is_rgb_mode:
            if self.pixels_per_word <= 0:
                raise ValueError(f"Virtual Display word size {self.word_size} is too small for RGB color depth {self.color_depth}.")
            return
        if not (1 <= self.color_depth <= self.INDEXED_MAX_COLOR_DEPTH):
            raise ValueError(f"Indexed Virtual Display color depth must be between 1 and {self.INDEXED_MAX_COLOR_DEPTH}.")
        if self.pixels_per_word <= 0:
            raise ValueError(f"Virtual Display word size {self.word_size} is too small for color depth {self.color_depth}.")
        if len(self.palette_rgb) < self.palette_entry_count:
            raise ValueError(f"Virtual Display palette needs {self.palette_entry_count} colors, got {len(self.palette_rgb)}.")


@dataclass
class Net:
    """A Net is a single continuous wire (including jumped crossroads/tunnels)."""

    id: int
    blocks: Set[Tuple[int, int]] = field(default_factory=set)
    readers: List["Gate"] = field(default_factory=list)
    writers: List["Gate"] = field(default_factory=list)


@dataclass
class Gate:
    """A Gate represents a logical component (can be multiple blocks)."""

    id: int
    kind: str
    blocks: Set[Tuple[int, int]] = field(default_factory=set)
    block_ids: Set[int] = field(default_factory=set)
    initial_state: int = 0
    inputs: List[Net] = field(default_factory=list)
    outputs: List[Net] = field(default_factory=list)
    is_virtual: bool = False
    vmem_bit_index: Optional[int] = None
    vmem_value_bit: Optional[int] = None

    def __hash__(self):
        return hash(self.id)


@dataclass
class BusComponent:
    """A contiguous bus body that can carry several independent logical nets."""

    id: int
    blocks: Set[Tuple[int, int]] = field(default_factory=set)
    net_ids: Set[int] = field(default_factory=set)


@dataclass
class Region:
    """A strongly connected gate region plus its graph boundaries."""

    id: int
    gates: List[Gate] = field(default_factory=list)
    input_nets: List[Net] = field(default_factory=list)
    output_nets: List[Net] = field(default_factory=list)
    downstream_region_ids: List[int] = field(default_factory=list)
    upstream_region_ids: List[int] = field(default_factory=list)
    is_volatile: bool = False


@dataclass
class VisualBlock:
    """A renderable block tied to either a gate output, a net, or neither."""

    x: int
    y: int
    block_id: int
    base_color: Optional[int] = None
    signal_kind: int = 0
    signal_id: int = 0
    style_flags: int = 0


class CircuitCompiler:
    VOLATILE_KINDS = {"CLOCK", "RANDOM"}
    VISUAL_SIGNAL_NONE = 0
    VISUAL_SIGNAL_GATE = 1
    VISUAL_SIGNAL_NET = 2
    VISUAL_SIGNAL_BUS = 3
    VISUAL_STYLE_VMEM = 1

    def __init__(
        self,
        grid: List[List[int]],
        config: BlockConfig,
        vmem_config: Optional[VMemConfig] = None,
        virtual_display_config: Optional[VirtualDisplayConfig] = None,
        clock_interval: int = 1,
    ):
        self.grid = grid
        self.config = config
        self.vmem_config = vmem_config
        self.virtual_display_config = virtual_display_config
        self.clock_interval = max(1, int(clock_interval))
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

        self.gates: Dict[int, Gate] = {}
        self.nets: Dict[int, Net] = {}
        self.bus_components: Dict[int, BusComponent] = {}

        self.coord_to_gate: Dict[Tuple[int, int], Gate] = {}
        self.coord_to_net: Dict[Tuple[int, int], Net] = {}
        self.coord_to_bus: Dict[Tuple[int, int], BusComponent] = {}
        self.bus_trace_contacts: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}
        self.bus_gate_contacts: Dict[int, Dict[str, Set[Tuple[int, int]]]] = {}

    def get_id_at(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return 0

    def compile(self):
        """Main compilation routine."""
        if self.vmem_config is not None:
            self.vmem_config.validate()
        if self.virtual_display_config is not None:
            self.virtual_display_config.validate(self.vmem_config)
        self._extract_buses()
        self._extract_gates()
        self._inject_vmem_gates()
        self._extract_nets()
        self._link_graph()
        return self.gates, self.nets

    def build_regions(self) -> List[Region]:
        """Collapse the gate dependency graph into SCC regions."""
        if not self.gates:
            return []

        gate_edges: Dict[int, Set[int]] = {gate_id: set() for gate_id in self.gates}
        reverse_edges: Dict[int, Set[int]] = {gate_id: set() for gate_id in self.gates}

        for net in self.nets.values():
            writer_ids = {gate.id for gate in net.writers}
            reader_ids = {gate.id for gate in net.readers}
            for writer_id in writer_ids:
                for reader_id in reader_ids:
                    gate_edges[writer_id].add(reader_id)
                    reverse_edges[reader_id].add(writer_id)

        components = self._tarjan_scc(gate_edges)
        component_index_by_gate: Dict[int, int] = {}
        for index, component in enumerate(components):
            for gate_id in component:
                component_index_by_gate[gate_id] = index

        region_edges: Dict[int, Set[int]] = {index: set() for index in range(len(components))}
        region_reverse_edges: Dict[int, Set[int]] = {index: set() for index in range(len(components))}
        for gate_id, targets in gate_edges.items():
            src_region = component_index_by_gate[gate_id]
            for target_id in targets:
                dst_region = component_index_by_gate[target_id]
                if src_region != dst_region:
                    region_edges[src_region].add(dst_region)
                    region_reverse_edges[dst_region].add(src_region)

        topo_component_order = self._topologically_sort_components(region_edges)
        region_id_by_component: Dict[int, int] = {component_index: region_id for region_id, component_index in enumerate(topo_component_order, start=1)}

        regions: Dict[int, Region] = {}
        for component_index in topo_component_order:
            component_gate_ids = sorted(components[component_index])
            gates = [self.gates[gate_id] for gate_id in component_gate_ids]
            gate_id_set = set(component_gate_ids)

            input_nets: Dict[int, Net] = {}
            output_nets: Dict[int, Net] = {}
            for gate in gates:
                for net in gate.inputs:
                    writer_ids = {writer.id for writer in net.writers}
                    if not writer_ids or not writer_ids.issubset(gate_id_set):
                        input_nets[net.id] = net
                for net in gate.outputs:
                    output_nets[net.id] = net

            region_id = region_id_by_component[component_index]
            regions[region_id] = Region(
                id=region_id,
                gates=gates,
                input_nets=[input_nets[net_id] for net_id in sorted(input_nets)],
                output_nets=[output_nets[net_id] for net_id in sorted(output_nets)],
                downstream_region_ids=sorted(region_id_by_component[target] for target in region_edges[component_index]),
                upstream_region_ids=sorted(region_id_by_component[source] for source in region_reverse_edges[component_index]),
                is_volatile=any(gate.kind in self.VOLATILE_KINDS for gate in gates),
            )

        return [regions[region_id] for region_id in sorted(regions)]

    def build_visual_blocks(self) -> List[VisualBlock]:
        """Export the occupied blocks and which logical signal should light each one."""
        visual_blocks: List[VisualBlock] = []

        for y in range(self.height):
            for x in range(self.width):
                block_id = self.get_id_at(x, y)
                if block_id in self.config.empty_ids:
                    continue

                signal_kind = self.VISUAL_SIGNAL_NONE
                signal_id = 0

                gate = self.coord_to_gate.get((x, y))
                if gate is not None:
                    signal_kind = self.VISUAL_SIGNAL_GATE
                    signal_id = gate.id
                else:
                    net = self.coord_to_net.get((x, y))
                    if net is not None:
                        signal_kind = self.VISUAL_SIGNAL_NET
                        signal_id = net.id
                    else:
                        bus_component = self.coord_to_bus.get((x, y))
                        if bus_component is not None:
                            signal_kind = self.VISUAL_SIGNAL_BUS
                            signal_id = bus_component.id

                visual_blocks.append(
                    VisualBlock(
                        x=x,
                        y=y,
                        block_id=block_id,
                        signal_kind=signal_kind,
                        signal_id=signal_id,
                    )
                )

        if self.vmem_config is not None and self.vmem_config.enabled:
            for gate in self._sorted_virtual_vmem_gates():
                base_color = self._vmem_base_color(gate)
                for block_x, block_y in sorted(gate.blocks):
                    visual_blocks.append(
                        VisualBlock(
                            x=block_x,
                            y=block_y,
                            block_id=0,
                            base_color=base_color,
                            signal_kind=self.VISUAL_SIGNAL_GATE,
                            signal_id=gate.id,
                            style_flags=self.VISUAL_STYLE_VMEM,
                        )
                    )

        return visual_blocks

    def _extract_buses(self):
        """Pass 0: Flood-fill contiguous bus bodies and index their touch points."""
        visited = set()
        bus_counter = 1

        for y in range(self.height):
            for x in range(self.width):
                block_id = self.get_id_at(x, y)
                if (x, y) in visited or not self.config.is_bus(block_id):
                    continue

                bus_blocks = set()
                queue = collections.deque([(x, y)])
                visited.add((x, y))

                while queue:
                    cx, cy = queue.popleft()
                    bus_blocks.add((cx, cy))

                    for direction in Direction:
                        next_pos = self._bus_neighbor(cx, cy, direction)
                        if next_pos is None or next_pos in visited:
                            continue
                        visited.add(next_pos)
                        queue.append(next_pos)

                component = BusComponent(id=bus_counter, blocks=bus_blocks)
                self.bus_components[bus_counter] = component
                for bx, by in bus_blocks:
                    self.coord_to_bus[(bx, by)] = component
                bus_counter += 1

        for component in self.bus_components.values():
            trace_contacts: Dict[int, Set[Tuple[int, int]]] = {}
            gate_contacts: Dict[str, Set[Tuple[int, int]]] = {}
            for bx, by in component.blocks:
                for direction in Direction:
                    nx, ny = bx + direction.value[0], by + direction.value[1]
                    neighbor_id = self.get_id_at(nx, ny)
                    if self.config.is_trace_wire(neighbor_id):
                        trace_contacts.setdefault(neighbor_id, set()).add((nx, ny))
                    elif neighbor_id in self.config.crossroad_ids:
                        jump_x, jump_y = nx + direction.value[0], ny + direction.value[1]
                        jump_id = self.get_id_at(jump_x, jump_y)
                        if self.config.is_trace_wire(jump_id):
                            trace_contacts.setdefault(jump_id, set()).add((jump_x, jump_y))
                    gate_kind = self.config.get_gate_kind(neighbor_id)
                    if gate_kind is not None:
                        gate_contacts.setdefault(gate_kind, set()).add((nx, ny))
            self.bus_trace_contacts[component.id] = trace_contacts
            self.bus_gate_contacts[component.id] = gate_contacts

    def _bus_neighbor(
        self,
        x: int,
        y: int,
        direction: Direction,
    ) -> Optional[Tuple[int, int]]:
        return self._straight_link_neighbor(
            x,
            y,
            direction,
            self.config.is_bus,
            self._bus_tunnel_signature,
        )

    def _extract_gates(self):
        """Pass 1: Flood-fill to find all multi-block gates, including bus-linked tiles."""
        visited = set()
        gate_counter = 1

        for y in range(self.height):
            for x in range(self.width):
                block_id = self.get_id_at(x, y)
                gate_kind = self.config.get_gate_kind(block_id)
                if (x, y) in visited or gate_kind is None:
                    continue

                def is_same_gate_kind(block_id: int, expected_kind: str = gate_kind) -> bool:
                    return self.config.get_gate_kind(block_id) == expected_kind

                gate_blocks = set()
                gate_block_ids = set()
                queue = collections.deque([(x, y)])
                visited.add((x, y))

                while queue:
                    cx, cy = queue.popleft()
                    current_block_id = self.get_id_at(cx, cy)
                    gate_blocks.add((cx, cy))
                    gate_block_ids.add(current_block_id)

                    for direction in Direction:
                        nx, ny = cx + direction.value[0], cy + direction.value[1]
                        next_block_id = self.get_id_at(nx, ny)
                        gate_neighbor = self._straight_link_neighbor(
                            cx,
                            cy,
                            direction,
                            is_same_gate_kind,
                            self._gate_tunnel_signature,
                        )
                        if gate_neighbor is not None:
                            if gate_neighbor not in visited:
                                visited.add(gate_neighbor)
                                queue.append(gate_neighbor)
                            continue

                        if self.config.is_bus(next_block_id):
                            bus_component = self.coord_to_bus.get((nx, ny))
                            if bus_component is None:
                                continue

                            for touch_x, touch_y in sorted(self.bus_gate_contacts.get(bus_component.id, {}).get(gate_kind, set())):
                                if (touch_x, touch_y) in visited:
                                    continue
                                visited.add((touch_x, touch_y))
                                queue.append((touch_x, touch_y))
                            continue

                new_gate = Gate(
                    id=gate_counter,
                    kind=gate_kind,
                    blocks=gate_blocks,
                    block_ids=gate_block_ids,
                    initial_state=self._resolve_initial_state(gate_block_ids),
                )
                self.gates[gate_counter] = new_gate
                for bx, by in gate_blocks:
                    self.coord_to_gate[(bx, by)] = new_gate
                gate_counter += 1

    def _inject_vmem_gates(self) -> None:
        if self.vmem_config is None or not self.vmem_config.enabled:
            return

        next_gate_id = (max(self.gates) + 1) if self.gates else 1
        for gate_kind, layout in (
            (VMEM_ADDRESS_KIND, self.vmem_config.address_layout),
            (VMEM_DATA_KIND, self.vmem_config.data_layout),
        ):
            if layout is None:
                continue
            placement_layout = replace(layout, offset_x=-layout.offset_x)
            for bit_index in range(layout.bits):
                placement_bit_index = layout.bits - 1 - bit_index
                gate_blocks = placement_layout.bit_blocks(placement_bit_index)
                self._validate_vmem_gate_blocks(gate_kind, bit_index, gate_blocks)
                gate = Gate(
                    id=next_gate_id,
                    kind=gate_kind,
                    blocks=gate_blocks,
                    initial_state=0,
                    is_virtual=True,
                    vmem_bit_index=bit_index,
                    vmem_value_bit=(layout.bits - 1 - bit_index),
                )
                self.gates[next_gate_id] = gate
                for block_x, block_y in gate_blocks:
                    self.coord_to_gate[(block_x, block_y)] = gate
                next_gate_id += 1

    def _validate_vmem_gate_blocks(
        self,
        gate_kind: str,
        bit_index: int,
        gate_blocks: Set[Tuple[int, int]],
    ) -> None:
        for block_x, block_y in gate_blocks:
            if not (0 <= block_x < self.width and 0 <= block_y < self.height):
                raise ValueError(f"{gate_kind} bit {bit_index} block ({block_x}, {block_y}) is outside the project bounds.")
            if self.get_id_at(block_x, block_y) not in self.config.empty_ids:
                raise ValueError(f"{gate_kind} bit {bit_index} overlaps a non-empty block at ({block_x}, {block_y}).")
            if (block_x, block_y) in self.coord_to_gate:
                raise ValueError(f"{gate_kind} bit {bit_index} overlaps another gate at ({block_x}, {block_y}).")

    def _extract_nets(self):
        """Pass 2: Trace logical nets across wires, with buses acting as multiplexed carriers."""
        visited_wire_coords = set()
        visited_bus_lanes = set()
        net_counter = 1

        for y in range(self.height):
            for x in range(self.width):
                block_id = self.get_id_at(x, y)
                if (x, y) in visited_wire_coords or not self.config.is_trace_wire(block_id):
                    continue

                current_net = Net(id=net_counter)
                self.nets[net_counter] = current_net

                queue = collections.deque([(x, y)])
                visited_wire_coords.add((x, y))

                while queue:
                    cx, cy = queue.popleft()
                    current_block_id = self.get_id_at(cx, cy)
                    current_net.blocks.add((cx, cy))
                    self.coord_to_net[(cx, cy)] = current_net

                    for direction in Direction:
                        trace_neighbor = self._straight_link_neighbor(
                            cx,
                            cy,
                            direction,
                            self.config.is_trace_wire,
                            self._trace_tunnel_signature,
                        )
                        if trace_neighbor is not None:
                            if trace_neighbor not in visited_wire_coords:
                                visited_wire_coords.add(trace_neighbor)
                                queue.append(trace_neighbor)
                            continue

                        bus_neighbor = self._straight_link_neighbor(cx, cy, direction, self.config.is_bus)
                        if bus_neighbor is not None:
                            self._queue_bus_lane_contacts(
                                bus_neighbor[0],
                                bus_neighbor[1],
                                current_block_id,
                                current_net,
                                queue,
                                visited_wire_coords,
                                visited_bus_lanes,
                            )

                net_counter += 1

    def _straight_link_neighbor(
        self,
        x: int,
        y: int,
        direction: Direction,
        accepts_block: Callable[[int], bool],
        tunnel_signature_resolver: Optional[Callable[[int], Optional[TunnelSignature]]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Return the accepted neighbor reached straight ahead.

        Traces, buses, and gates share the same physical linking rule:
        connect to an adjacent accepted block, jump straight over a crossroad,
        or jump through a tunnel whose far side has the caller's signature.
        """
        nx, ny = x + direction.value[0], y + direction.value[1]
        neighbor_id = self.get_id_at(nx, ny)
        if accepts_block(neighbor_id):
            return (nx, ny)

        if neighbor_id in self.config.crossroad_ids:
            jump_x, jump_y = nx + direction.value[0], ny + direction.value[1]
            if accepts_block(self.get_id_at(jump_x, jump_y)):
                return (jump_x, jump_y)
            return None

        if neighbor_id in self.config.tunnel_ids and tunnel_signature_resolver is not None:
            source_block_id = self.get_id_at(x, y)
            expected_signature = tunnel_signature_resolver(source_block_id)
            if expected_signature is None:
                return None
            return self._find_tunnel_exit(
                start_x=nx,
                start_y=ny,
                direction=direction,
                expected_signature=expected_signature,
                signature_resolver=tunnel_signature_resolver,
                source_x=x,
                source_y=y,
                source_block_id=source_block_id,
            )

        return None

    def _find_tunnel_exit(
        self,
        start_x: int,
        start_y: int,
        direction: Direction,
        expected_signature: TunnelSignature,
        signature_resolver: Callable[[int], Optional[TunnelSignature]],
        source_x: int,
        source_y: int,
        source_block_id: int,
    ) -> Tuple[int, int]:
        """Raycast to find the first tunnel whose far-side block matches the source signature."""
        rx, ry = start_x + direction.value[0], start_y + direction.value[1]

        while 0 <= rx < self.width and 0 <= ry < self.height:
            check_id = self.get_id_at(rx, ry)

            if check_id in self.config.tunnel_ids:
                exit_wire_x = rx + direction.value[0]
                exit_wire_y = ry + direction.value[1]
                if signature_resolver(self.get_id_at(exit_wire_x, exit_wire_y)) == expected_signature:
                    return (exit_wire_x, exit_wire_y)

            rx += direction.value[0]
            ry += direction.value[1]

        raise ValueError(f"Tunnel at ({start_x}, {start_y}) looking {direction.name} from {self._describe_block(source_block_id)} at ({source_x}, {source_y}) could not find a matching tunnel exit.")

    def _trace_tunnel_signature(self, block_id: int) -> Optional[TunnelSignature]:
        if self.config.is_trace_wire(block_id):
            return ("trace", block_id)
        return None

    def _bus_tunnel_signature(self, block_id: int) -> Optional[TunnelSignature]:
        if self.config.is_bus(block_id):
            return ("bus", block_id)
        return None

    def _gate_tunnel_signature(self, block_id: int) -> Optional[TunnelSignature]:
        gate_kind = self.config.get_gate_kind(block_id)
        if gate_kind is None:
            return None
        return ("gate", gate_kind)

    def _describe_block(self, block_id: int) -> str:
        gate_kind = self.config.get_gate_kind(block_id)
        if gate_kind is not None:
            return f"{gate_kind} gate"
        if self.config.is_trace_wire(block_id):
            return f"trace block 0x{block_id:08X}"
        if self.config.is_bus(block_id):
            return f"bus block 0x{block_id:08X}"
        if block_id in self.config.tunnel_ids:
            return f"tunnel block 0x{block_id:08X}"
        if block_id in self.config.crossroad_ids:
            return f"crossroad block 0x{block_id:08X}"
        if block_id in self.config.empty_ids:
            return "empty space"
        return f"block 0x{block_id:08X}"

    def _queue_bus_lane_contacts(
        self,
        bus_x: int,
        bus_y: int,
        lane_block_id: int,
        current_net: Net,
        queue: collections.deque[Tuple[int, int]],
        visited_wire_coords: Set[Tuple[int, int]],
        visited_bus_lanes: Set[Tuple[int, int]],
    ) -> None:
        bus_component = self.coord_to_bus.get((bus_x, bus_y))
        if bus_component is None:
            return

        lane_key = (bus_component.id, lane_block_id)
        if lane_key in visited_bus_lanes:
            return

        visited_bus_lanes.add(lane_key)
        bus_component.net_ids.add(current_net.id)

        for touch_x, touch_y in sorted(self.bus_trace_contacts.get(bus_component.id, {}).get(lane_block_id, set())):
            if (touch_x, touch_y) in visited_wire_coords:
                continue
            visited_wire_coords.add((touch_x, touch_y))
            queue.append((touch_x, touch_y))

    def _link_graph(self):
        """Pass 3: Connect Gates and Nets together based on Read/Write blocks."""
        for gate in self.gates.values():
            for gx, gy in gate.blocks:
                for direction in Direction:
                    nx, ny = gx + direction.value[0], gy + direction.value[1]
                    adj_id = self.get_id_at(nx, ny)

                    if (nx, ny) not in self.coord_to_net:
                        continue

                    target_net = self.coord_to_net[(nx, ny)]
                    if adj_id in self.config.read_ids:
                        if target_net not in gate.inputs:
                            gate.inputs.append(target_net)
                        if gate not in target_net.readers:
                            target_net.readers.append(gate)
                    elif adj_id in self.config.write_ids:
                        if target_net not in gate.outputs:
                            gate.outputs.append(target_net)
                        if gate not in target_net.writers:
                            target_net.writers.append(gate)

    def _tarjan_scc(self, edges: Dict[int, Set[int]]) -> List[List[int]]:
        index_counter = 0
        indices: Dict[int, int] = {}
        lowlinks: Dict[int, int] = {}
        stack: List[int] = []
        on_stack: Set[int] = set()
        components: List[List[int]] = []

        def strongconnect(node: int):
            nonlocal index_counter
            indices[node] = index_counter
            lowlinks[node] = index_counter
            index_counter += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in sorted(edges[node]):
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                component: List[int] = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)

        for node in sorted(edges):
            if node not in indices:
                strongconnect(node)

        return components

    def _topologically_sort_components(self, edges: Dict[int, Set[int]]) -> List[int]:
        indegree = {node: 0 for node in edges}
        for targets in edges.values():
            for target in targets:
                indegree[target] += 1

        ready = collections.deque(sorted(node for node, degree in indegree.items() if degree == 0))
        order: List[int] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for neighbor in sorted(edges[node]):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    ready.append(neighbor)

        return order

    def _resolve_initial_state(self, block_ids: Set[int]) -> int:
        if not block_ids:
            return 0
        weighted_state = sum(self.config.gate_initial_states.get(block_id, 0) for block_id in block_ids)
        return 1 if weighted_state * 2 > len(block_ids) else 0

    def _sorted_virtual_vmem_gates(self) -> List[Gate]:
        return [gate for gate in self._sorted_gates() if gate.kind in {VMEM_ADDRESS_KIND, VMEM_DATA_KIND}]

    def _sorted_gates(self) -> List[Gate]:
        return [self.gates[gate_id] for gate_id in sorted(self.gates)]

    def _vmem_base_color(self, gate: Gate) -> int:
        if gate.vmem_bit_index is None:
            return 0xFFFFFFFF
        total_bits = self.vmem_config.address_bits if gate.kind == VMEM_ADDRESS_KIND else self.vmem_config.data_bits
        return self._vmem_gradient_color(gate.vmem_bit_index, total_bits)

    @staticmethod
    def _vmem_gradient_color(bit_index: int, total_bits: int) -> int:
        gradient = (
            (255, 217, 102),
            (255, 176, 92),
            (255, 111, 119),
            (235, 83, 177),
            (152, 102, 255),
            (102, 166, 255),
        )
        if total_bits <= 1:
            red, green, blue = gradient[0]
            return CircuitCompiler._pack_rgba(red, green, blue)

        position = bit_index / (total_bits - 1)
        scaled = position * (len(gradient) - 1)
        left_index = int(scaled)
        right_index = min(left_index + 1, len(gradient) - 1)
        blend = scaled - left_index
        left = gradient[left_index]
        right = gradient[right_index]
        red = round(left[0] + ((right[0] - left[0]) * blend))
        green = round(left[1] + ((right[1] - left[1]) * blend))
        blue = round(left[2] + ((right[2] - left[2]) * blend))
        return CircuitCompiler._pack_rgba(red, green, blue)

    @staticmethod
    def _pack_rgba(red: int, green: int, blue: int) -> int:
        return (red & 0xFF) | ((green & 0xFF) << 8) | ((blue & 0xFF) << 16) | 0xFF000000
