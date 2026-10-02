# VCB simulator

[![CI](https://github.com/RomanPolek/vcb-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/RomanPolek/vcb-simulator/actions/workflows/ci.yml)

A simulator for Virtual Circuit Board `.vcb` projects, with a native C event
engine and an interactive pygame circuit and virtual-display viewer.

## Setup and run

Requires Python 3.11+, GCC on PATH, and the packages in `requirements.txt`.
On Windows, use a GCC installation that targets the same architecture as Python.

```sh
python -m pip install -r requirements.txt
python main.py --project "path/to/project.vcb"
```

Try an included test circuit with `python main.py --project fixtures/test2.vcb`.
Use `--compiler "path/to/gcc.exe"` to choose GCC explicitly.
`--project` is required. Running `python main.py` shows usage and a missing-project
message. Use `python main.py --help` for all options and an example.

The engine compiles on first use with `-O3 -march=native`. Generated libraries
are cached in `output/`, which can be deleted when the simulator is closed.
Simulation runs on its own thread at maximum speed, publishing snapshots to
the viewer at roughly 60 FPS (depending on circuit size and hardware). The window
title reports measured ticks per second.

Controls: Space pauses; period steps while paused; R resets; mouse wheel zooms;
drag pans; F fits the circuit; Escape closes the window.

For a short run without a window:

```sh
python main.py --project fixtures/test2.vcb --benchmark 3
```

## Vmem and assembly helpers

This project also contains project memory loading and inline/external assembly:

- `compiler.extractor.load_project(path)` reads a project.
- `compiler.extractor.extract_vmem_from_project(project)` decodes stored memory.
- `compiler.assembler.load_project_assembly_source(project, path)` reads inline
  assembly or an external source relative to the project file.
- `compiler.assembler.assemble_source_words(source)` returns address/value pairs.
- `main.build_initial_vmem_image(project, path)` loads stored memory, then overlays
  assembled words, producing the full big-endian memory image.
- `main.build_vmem_config(project, path)` combines that image with project layouts
  and persistence settings for the runtime.

For example:

```python
from compiler.assembler import assemble_source_words

words = assemble_source_words("origin 0x10\n0x12345678")
assert words == {0x10: 0x12345678}
```

Vmem contains 2^20 32-bit words. An enabled project's stored image must be exactly
4 MiB when present. Assembly requires vmem to be enabled. Reset reloads the
initial memory image.

## Development

```sh
python -m unittest discover -s tests
```

Tests cover circuit logic, vmem, native/reference equivalence, CLI validation,
and rendering. Native tests require GCC. See [CONTRIBUTING.md](CONTRIBUTING.md).

The source is limited to `compiler/`, `runner/`, the entry point and settings,
plus regression tests and their circuit fixtures. `runner/simulation.py` supplies
the native engine's execution plan, snapshot/worker types, and Python reference.

## License

[MIT](LICENSE), using the [standard MIT license](https://opensource.org/license/mit).
Third-party packages and independently supplied circuit projects retain their
own licenses. This is an independent project, not an official Virtual Circuit
Board release.
