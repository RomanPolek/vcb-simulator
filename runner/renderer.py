from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from runner.simulation import CircuitRuntime, RuntimeSnapshot, SimulationDriver

VISUAL_SIGNAL_GATE = 1
VISUAL_SIGNAL_NET = 2
VISUAL_SIGNAL_BUS = 3
BACKGROUND_RGB = (0x18, 0x20, 0x2A)
MIN_ZOOM = 0.05
MAX_ZOOM = 64.0


def _clamp_channel(value: float) -> int:
    if value <= 0.0:
        return 0
    rounded = int(value + 0.5)
    return min(255, rounded)


def _shade_rgb(color: int, factor_256: int) -> int:
    rb = color & 0xFF00FF
    green = color & 0x00FF00
    return (((rb * factor_256) >> 8) & 0xFF00FF) | (((green * factor_256) >> 8) & 0x00FF00)


def _scale_rgb(color: int, scale: float) -> int:
    red = (color >> 16) & 0xFF
    green = (color >> 8) & 0xFF
    blue = color & 0xFF
    return (_clamp_channel(red * scale) << 16) | (_clamp_channel(green * scale) << 8) | _clamp_channel(blue * scale)


def _rgb_tuple(color: int) -> Tuple[int, int, int]:
    return ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)


def _apply_activity_to_packed_rgba(packed_rgba: int, active: bool) -> int:
    red = float(packed_rgba & 0xFF)
    green = float((packed_rgba >> 8) & 0xFF)
    blue = float((packed_rgba >> 16) & 0xFF)
    luma = ((0.2126 * red) + (0.7152 * green) + (0.0722 * blue)) / 255.0

    if packed_rgba == 0xFFFFFFFF:
        return 0xFFFFFF if active else 0x5F6870

    if active:
        scale = 1.08 + ((1.0 - luma) * 0.42)
        out_red = _clamp_channel((red * scale) + 14.0)
        out_green = _clamp_channel((green * scale) + 14.0)
        out_blue = _clamp_channel((blue * scale) + 14.0)
    else:
        out_red = _clamp_channel((red * 0.46) + 12.0)
        out_green = _clamp_channel((green * 0.46) + 12.0)
        out_blue = _clamp_channel((blue * 0.46) + 12.0)
        if luma < 0.18:
            out_red = _clamp_channel(float(out_red) + 7.0)
            out_green = _clamp_channel(float(out_green) + 7.0)
            out_blue = _clamp_channel(float(out_blue) + 7.0)

    return (out_red << 16) | (out_green << 8) | out_blue


@dataclass(frozen=True)
class _SignalLookup:
    block_indices: np.ndarray
    signal_ids: np.ndarray


class VisualScene:
    def __init__(self, runtime: CircuitRuntime):
        self.runtime = runtime
        self.display = runtime.virtual_display_config
        self.min_x, self.min_y, self.max_x, self.max_y = self._bounds()
        self.width = max(1, (self.max_x - self.min_x) + 1)
        self.height = max(1, (self.max_y - self.min_y) + 1)

        blocks = list(runtime.visual_blocks)
        count = len(blocks)
        self.xs = np.empty(count, dtype=np.int32)
        self.ys = np.empty(count, dtype=np.int32)
        self.signal_kinds = np.empty(count, dtype=np.uint8)
        self.signal_ids = np.empty(count, dtype=np.uint32)
        self.inactive_colors = np.empty(count, dtype=np.uint32)
        self.active_colors = np.empty(count, dtype=np.uint32)

        for index, block in enumerate(blocks):
            packed = block.base_color if block.base_color is not None else block.block_id
            self.xs[index] = block.x - self.min_x
            self.ys[index] = block.y - self.min_y
            self.signal_kinds[index] = block.signal_kind
            self.signal_ids[index] = block.signal_id
            self.inactive_colors[index] = _apply_activity_to_packed_rgba(packed, False)
            self.active_colors[index] = _apply_activity_to_packed_rgba(packed, True)

        # Surface pixels are 0xRRGGBB, with x contiguous in memory. Matching
        # that layout avoids per-pixel RGB conversion and a strided transpose.
        self.base_frame = np.empty((self.width, self.height), dtype=np.uint32, order="F")
        self.base_frame[:, :] = (BACKGROUND_RGB[0] << 16) | (BACKGROUND_RGB[1] << 8) | BACKGROUND_RGB[2]
        if count:
            self.base_frame[self.xs, self.ys] = self.inactive_colors

        self.gate_lookup = self._lookup(VISUAL_SIGNAL_GATE, runtime.gate_count + 1)
        self.net_lookup = self._lookup(VISUAL_SIGNAL_NET, runtime.net_count + 1)
        self.bus_lookup = self._lookup(VISUAL_SIGNAL_BUS, runtime.bus_count + 1)
        self.vdisplay_inactive = _scale_rgb(0x101418, 1.45)
        self._prepare_virtual_display()

    def render_frame(self, snapshot: RuntimeSnapshot) -> np.ndarray:
        frame = self.base_frame.copy(order="F")
        if len(self.signal_ids):
            active = np.zeros(len(self.signal_ids), dtype=bool)
            self._mark_active(active, self.gate_lookup, snapshot.gate_out)
            self._mark_active(active, self.net_lookup, snapshot.net_state)
            self._mark_active(active, self.bus_lookup, snapshot.bus_state)
            if active.any():
                frame[self.xs[active], self.ys[active]] = self.active_colors[active]
        self._render_virtual_display(frame, snapshot)
        return frame

    def _bounds(self) -> Tuple[int, int, int, int]:
        xs = [block.x for block in self.runtime.visual_blocks]
        ys = [block.y for block in self.runtime.visual_blocks]

        if self.display.enabled:
            xs.extend(
                [
                    self.display.position_x,
                    self.display.position_x + (self.display.resolution_x * self.display.scale_x) - 1,
                ]
            )
            ys.extend(
                [
                    self.display.position_y,
                    self.display.position_y + (self.display.resolution_y * self.display.scale_y) - 1,
                ]
            )

        if not xs or not ys:
            return (0, 0, max(0, self.runtime.grid_width - 1), max(0, self.runtime.grid_height - 1))
        return (min(xs), min(ys), max(xs), max(ys))

    def _lookup(self, signal_kind: int, capacity: int) -> _SignalLookup:
        mask = (self.signal_kinds == signal_kind) & (self.signal_ids < capacity)
        indices = np.nonzero(mask)[0]
        return _SignalLookup(indices, self.signal_ids[indices].astype(np.intp, copy=False))

    @staticmethod
    def _mark_active(active: np.ndarray, lookup: _SignalLookup, state: bytes) -> None:
        if lookup.block_indices.size == 0:
            return
        state_array = np.frombuffer(state, dtype=np.uint8)
        active[lookup.block_indices] = state_array[lookup.signal_ids] != 0

    def _render_virtual_display(self, frame: np.ndarray, snapshot: RuntimeSnapshot) -> None:
        display = self.display
        if not display.enabled or not snapshot.vdisplay_words:
            return

        if self.vdisplay_total_pixels <= 0 or self.vdisplay_pixels_per_word <= 0:
            return
        if self.vdisplay_left >= self.vdisplay_right or self.vdisplay_top >= self.vdisplay_bottom:
            return

        words = np.asarray(snapshot.vdisplay_words, dtype=np.uint32)
        if words.size == 0:
            return

        if self.vdisplay_is_rgb:
            color_values = words[self.vdisplay_word_indices] & 0xFFFFFF
            colors = color_values
            inactive = color_values == 0
            if inactive.any():
                colors[inactive] = self.vdisplay_inactive
        else:
            palette_indices = (words[self.vdisplay_word_indices] >> self.vdisplay_shifts) & self.vdisplay_color_mask
            palette_indices = np.minimum(palette_indices, self.vdisplay_palette_limit)
            colors = self.vdisplay_palette[palette_indices]

        if display.direction == 0:
            pixels = colors.reshape((display.resolution_y, display.resolution_x))
        else:
            pixels = colors.reshape((display.resolution_x, display.resolution_y)).swapaxes(0, 1)

        if display.scale_y != 1:
            pixels = np.repeat(pixels, display.scale_y, axis=0)
        if display.scale_x != 1:
            pixels = np.repeat(pixels, display.scale_x, axis=1)

        frame[self.vdisplay_left : self.vdisplay_right, self.vdisplay_top : self.vdisplay_bottom] = pixels[
            self.vdisplay_crop_top : self.vdisplay_crop_bottom,
            self.vdisplay_crop_left : self.vdisplay_crop_right,
        ].swapaxes(0, 1)

    def _prepare_virtual_display(self) -> None:
        display = self.display
        self.vdisplay_total_pixels = 0
        self.vdisplay_pixels_per_word = 0
        self.vdisplay_word_indices = np.empty(0, dtype=np.intp)
        self.vdisplay_shifts = np.empty(0, dtype=np.uint32)
        self.vdisplay_color_mask = 0
        self.vdisplay_is_rgb = False
        self.vdisplay_palette_limit = 0
        self.vdisplay_palette = np.array([self.vdisplay_inactive], dtype=np.uint32)
        self.vdisplay_left = 0
        self.vdisplay_top = 0
        self.vdisplay_right = 0
        self.vdisplay_bottom = 0
        self.vdisplay_crop_left = 0
        self.vdisplay_crop_top = 0
        self.vdisplay_crop_right = 0
        self.vdisplay_crop_bottom = 0

        if not display.enabled:
            return
        total_pixels = display.resolution_x * display.resolution_y
        pixels_per_word = display.pixels_per_word
        if total_pixels <= 0 or pixels_per_word <= 0:
            return

        self.vdisplay_total_pixels = total_pixels
        self.vdisplay_pixels_per_word = pixels_per_word
        pixel_indices = np.arange(total_pixels, dtype=np.uint32)
        self.vdisplay_word_indices = (pixel_indices // pixels_per_word).astype(np.intp, copy=False)
        slot_indices = pixel_indices % pixels_per_word
        self.vdisplay_is_rgb = display.color_depth == display.RGB_COLOR_DEPTH
        self.vdisplay_color_mask = 0xFFFFFF if self.vdisplay_is_rgb else ((1 << display.color_depth) - 1)
        if not self.vdisplay_is_rgb:
            self.vdisplay_shifts = (display.word_size - ((slot_indices + 1) * display.color_depth)).astype(np.uint32, copy=False)
            palette = np.empty(max(1, display.palette_entry_count + 1), dtype=np.uint32)
            palette[:] = self.vdisplay_inactive
            for index, color in enumerate(display.palette_rgb[: display.palette_entry_count]):
                palette[index] = color if index != 0 else self.vdisplay_inactive
            self.vdisplay_palette = palette
            self.vdisplay_palette_limit = len(palette) - 1

        world_left = display.position_x - self.min_x
        world_top = display.position_y - self.min_y
        world_right = world_left + (display.resolution_x * display.scale_x)
        world_bottom = world_top + (display.resolution_y * display.scale_y)
        self.vdisplay_left = max(0, world_left)
        self.vdisplay_top = max(0, world_top)
        self.vdisplay_right = min(self.width, world_right)
        self.vdisplay_bottom = min(self.height, world_bottom)
        self.vdisplay_crop_left = self.vdisplay_left - world_left
        self.vdisplay_crop_top = self.vdisplay_top - world_top
        self.vdisplay_crop_right = self.vdisplay_crop_left + (self.vdisplay_right - self.vdisplay_left)
        self.vdisplay_crop_bottom = self.vdisplay_crop_top + (self.vdisplay_bottom - self.vdisplay_top)


class PygameRenderer:
    def __init__(
        self,
        runtime: CircuitRuntime,
        driver: SimulationDriver,
    ):
        self.runtime = runtime
        self.driver = driver
        self.scene = VisualScene(runtime)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.last_mouse = (0, 0)
        self.screen: pygame.Surface | None = None
        self.scene_surface: pygame.Surface | None = None

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("VCB Simulator")
        self.screen = pygame.display.set_mode((1600, 900), pygame.RESIZABLE)
        self._fit_view(*self.screen.get_size())

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self._handle_event(event)

            snapshot = self.driver.snapshot()
            self._draw(snapshot)
            self._update_title(snapshot)
            pygame.display.flip()
            clock.tick(self.driver.frame_hz)

        pygame.quit()

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE and self.screen is not None:
            self.screen = pygame.display.set_mode((max(1, event.w), max(1, event.h)), pygame.RESIZABLE)
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button in {1, 2, 3}:
            self.panning = True
            self.last_mouse = event.pos
            return

        if event.type == pygame.MOUSEBUTTONUP and event.button in {1, 2, 3}:
            self.panning = False
            return

        if event.type == pygame.MOUSEMOTION and self.panning:
            x, y = event.pos
            last_x, last_y = self.last_mouse
            self.pan_x += x - last_x
            self.pan_y += y - last_y
            self.last_mouse = event.pos
            return

        if event.type == pygame.MOUSEWHEEL:
            mouse = pygame.mouse.get_pos()
            self._zoom_at(mouse, 1.15**event.y)
            return

        if event.type != pygame.KEYDOWN:
            return

        if event.key in {pygame.K_ESCAPE, pygame.K_q}:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif event.key == pygame.K_SPACE:
            self.driver.toggle_pause()
        elif event.key in {pygame.K_PERIOD, pygame.K_PAGEUP}:
            self.driver.request_step()
        elif event.key == pygame.K_r:
            self.driver.request_reset()
        elif event.key == pygame.K_f and self.screen is not None:
            self._fit_view(*self.screen.get_size())
        elif event.key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self._zoom_at(pygame.mouse.get_pos(), 1.2)
        elif event.key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self._zoom_at(pygame.mouse.get_pos(), 1.0 / 1.2)

    def _draw(self, snapshot: RuntimeSnapshot) -> None:
        if self.screen is None:
            return
        screen_width, screen_height = self.screen.get_size()
        self.screen.fill(BACKGROUND_RGB)

        frame = self.scene.render_frame(snapshot)
        x0, y0, x1, y1 = self._visible_logical_rect(screen_width, screen_height)
        if x0 < x1 and y0 < y1:
            scene_surface = self._scene_surface()
            pixels = pygame.surfarray.pixels2d(scene_surface)
            np.copyto(pixels, frame)
            del pixels  # Release the surface lock before scaling/blitting.
            surface = scene_surface.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0))
            dest_left = int(math.floor(self.pan_x + (x0 * self.zoom)))
            dest_top = int(math.floor(self.pan_y + (y0 * self.zoom)))
            dest_width = max(1, int(math.ceil((x1 - x0) * self.zoom)))
            dest_height = max(1, int(math.ceil((y1 - y0) * self.zoom)))
            if dest_width != surface.get_width() or dest_height != surface.get_height():
                surface = pygame.transform.scale(surface, (dest_width, dest_height))
            self.screen.blit(surface, (dest_left, dest_top))


    def _scene_surface(self) -> pygame.Surface:
        if self.scene_surface is None:
            self.scene_surface = pygame.Surface((self.scene.width, self.scene.height), depth=32, masks=(0xFF0000, 0x00FF00, 0x0000FF, 0))
        return self.scene_surface

    def _visible_logical_rect(self, screen_width: int, screen_height: int) -> Tuple[int, int, int, int]:
        x0 = max(0, int(math.floor(-self.pan_x / self.zoom)))
        y0 = max(0, int(math.floor(-self.pan_y / self.zoom)))
        x1 = min(self.scene.width, int(math.ceil((screen_width - self.pan_x) / self.zoom)))
        y1 = min(self.scene.height, int(math.ceil((screen_height - self.pan_y) / self.zoom)))
        return x0, y0, x1, y1

    def _update_title(self, snapshot: RuntimeSnapshot) -> None:
        paused = " | paused" if self.driver.paused else ""
        mode = "max" if self.driver.max_mode else f"{self.driver.ticks_per_second or 0} TPS"
        pygame.display.set_caption(f"VCB Simulator | tick {snapshot.tick} | {self.driver.last_tps:,.0f} TPS | {mode}{paused}")

    def _fit_view(self, screen_width: int, screen_height: int) -> None:
        content_width = float(self.scene.width)
        content_height = float(self.scene.height)
        if content_width <= 0.0 or content_height <= 0.0:
            return
        self.zoom = min(screen_width / content_width, screen_height / content_height)
        self.zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom))
        self.pan_x = (screen_width - (content_width * self.zoom)) * 0.5
        self.pan_y = (screen_height - (content_height * self.zoom)) * 0.5

    def _zoom_at(self, mouse_pos: Tuple[int, int], scale: float) -> None:
        old_zoom = self.zoom
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, old_zoom * scale))
        if new_zoom == old_zoom:
            return
        mouse_x, mouse_y = mouse_pos
        world_x = (mouse_x - self.pan_x) / old_zoom
        world_y = (mouse_y - self.pan_y) / old_zoom
        self.zoom = new_zoom
        self.pan_x = mouse_x - (world_x * new_zoom)
        self.pan_y = mouse_y - (world_y * new_zoom)
