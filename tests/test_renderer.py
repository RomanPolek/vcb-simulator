"""Pixel-format regression tests: signals and virtual display remain identical."""
import unittest
from types import SimpleNamespace

import numpy as np
import pygame

from compiler.compiler import VisualBlock, VirtualDisplayConfig
from runner.renderer import VisualScene, PygameRenderer


def runtime(display=None, blocks=()):
    return SimpleNamespace(visual_blocks=blocks, virtual_display_config=display or VirtualDisplayConfig(),
                           grid_width=8, grid_height=8, gate_count=1, net_count=1, bus_count=1)


def snapshot(words=(), active=0):
    return SimpleNamespace(gate_out=bytes((0,active)), net_state=bytes((0,active)),
                           bus_state=bytes((0,active)), vdisplay_words=tuple(words))


class RendererTests(unittest.TestCase):
    def test_all_signal_types_switch_and_frames_are_independent(self):
        blocks=[VisualBlock(i,0,0xFFFFFFFF,signal_kind=i+1,signal_id=1) for i in range(3)]
        scene=VisualScene(runtime(blocks=blocks))
        off=scene.render_frame(snapshot())
        on=scene.render_frame(snapshot(active=1))
        np.testing.assert_array_equal(off[:,0], [0x5F6870]*3)
        np.testing.assert_array_equal(on[:,0], [0xFFFFFF]*3)
        np.testing.assert_array_equal(scene.render_frame(snapshot()),off)

    def test_display_pixels_and_surface_upload(self):
        inactive=0x171D23
        for depth in (2,24):
            for direction in (0,1):
                for scale in ((1,1),(2,3)):
                    with self.subTest(depth=depth,direction=direction,scale=scale):
                        display=VirtualDisplayConfig(enabled=True,position_x=-2,position_y=4,
                            resolution_x=3,resolution_y=2,scale_x=scale[0],scale_y=scale[1],
                            color_depth=depth,direction=direction,word_size=32 if depth==24 else 16,
                            palette_rgb=(0,0x112233,0x446688,0xAA77EE))
                        if depth==24:
                            words=[0,0xFF112233,0xFF446688,0xFFAA77EE,0xFF446688,0xFF112233]
                        else:
                            words=[sum(value << (14-2*i) for i,value in enumerate((0,1,2,3,2,1)))]
                        colors=[inactive,0x112233,0x446688,0xAA77EE,0x446688,0x112233]
                        expected=np.empty((3*scale[0],2*scale[1]),dtype=np.uint32)
                        for i,color in enumerate(colors):
                            x,y=(i%3,i//3) if direction==0 else (i//2,i%2)
                            expected[x*scale[0]:(x+1)*scale[0],y*scale[1]:(y+1)*scale[1]]=color
                        renderer=PygameRenderer(runtime(display),None)
                        renderer.screen=pygame.Surface((64,64))
                        renderer._fit_view(64,64)
                        state=snapshot(words)
                        np.testing.assert_array_equal(renderer.scene.render_frame(state),expected)
                        renderer._draw(state)
                        self.assertFalse(renderer._scene_surface().get_locked())
                        rgb=pygame.surfarray.array3d(renderer._scene_surface())
                        actual=(rgb[:,:,0].astype(np.uint32)<<16)|(rgb[:,:,1].astype(np.uint32)<<8)|rgb[:,:,2]
                        np.testing.assert_array_equal(actual,expected)
                        renderer._draw(state)  # Upload releases its surface view/lock.


if __name__=='__main__':
    unittest.main()
