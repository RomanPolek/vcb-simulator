from dataclasses import dataclass, field
from typing import Dict, List

from compiler.compiler import (
    BusComponent,
    Gate,
    Net,
    Region,
    VirtualDisplayConfig,
    VisualBlock,
    VMemConfig,
)


@dataclass(frozen=True)
class EmitterContext:
    gates: Dict[int, Gate]
    nets: Dict[int, Net]
    regions: List[Region]
    visual_blocks: List[VisualBlock]
    grid_width: int
    grid_height: int
    bus_components: Dict[int, BusComponent] = field(default_factory=dict)
    vmem_config: VMemConfig = field(default_factory=VMemConfig)
    virtual_display_config: VirtualDisplayConfig = field(default_factory=VirtualDisplayConfig)
    clock_interval: int = 1

    @classmethod
    def create(
        cls,
        gates: Dict[int, Gate],
        nets: Dict[int, Net],
        regions: List[Region],
        visual_blocks: List[VisualBlock],
        grid_width: int,
        grid_height: int,
        *,
        bus_components: Dict[int, BusComponent] | None = None,
        vmem_config: VMemConfig | None = None,
        virtual_display_config: VirtualDisplayConfig | None = None,
        clock_interval: int = 1,
    ) -> "EmitterContext":
        return cls(
            gates=gates,
            nets=nets,
            regions=regions,
            visual_blocks=visual_blocks,
            grid_width=grid_width,
            grid_height=grid_height,
            bus_components=bus_components or {},
            vmem_config=vmem_config or VMemConfig(),
            virtual_display_config=virtual_display_config or VirtualDisplayConfig(),
            clock_interval=max(1, int(clock_interval)),
        )

    @property
    def gate_count(self) -> int:
        return max(self.gates, default=0)

    @property
    def net_count(self) -> int:
        return max(self.nets, default=0)

    @property
    def bus_count(self) -> int:
        return max(self.bus_components, default=0)

    @property
    def vmem_enabled(self) -> bool:
        return self.vmem_config.enabled

    @property
    def virtual_display_enabled(self) -> bool:
        return self.virtual_display_config.enabled
