"""Stored memory and assembly must survive the same project-loading path."""
import base64
import tempfile
import unittest
from pathlib import Path

import zstandard

from compiler.assembler import assemble_source_words
from compiler.compiler import VMemConfig
from main import build_initial_vmem_image, build_vmem_config


class VMemLoadingTests(unittest.TestCase):
    def test_assembly_overlays_stored_words(self):
        stored = bytearray(VMemConfig.WORD_COUNT * 4)
        stored[4:12] = bytes.fromhex("11223344 55667788")
        project = {
            "vmem_data": base64.b64encode(zstandard.ZstdCompressor().compress(stored)).decode(),
            "assembly": "origin 1\n0xAABBCCDD",
        }
        image = build_initial_vmem_image(project, "example.vcb")
        self.assertEqual(len(image), len(stored))
        self.assertEqual(image[4:12], bytes.fromhex("AABBCCDD 55667788"))

    def test_external_assembly_resolves_relative_to_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "program.asm").write_text("origin 0x10\n0x12345678", encoding="utf-8")
            project = {"assembly_is_external": True, "assembly": "program.asm"}
            image = build_initial_vmem_image(project, root / "example.vcb")
            self.assertEqual(image[64:68], bytes.fromhex("12345678"))

    def test_rejects_wrong_stored_image_size(self):
        project = {"vmem_data": base64.b64encode(zstandard.ZstdCompressor().compress(b"abcd")).decode()}
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            build_initial_vmem_image(project, "example.vcb")

    def test_assembly_requires_enabled_memory(self):
        with self.assertRaisesRegex(ValueError, "VMem to be enabled"):
            build_vmem_config({"assembly": "1"}, "example.vcb")

    def test_standalone_assembly_helper(self):
        self.assertEqual(assemble_source_words("origin 0x10\n0x12345678"), {16: 0x12345678})
