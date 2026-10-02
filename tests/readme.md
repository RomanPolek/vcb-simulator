# Regression tests

Run `python -m unittest discover -s tests` from the repository root.

GCC is required for native builds. The native tests compare gate/net/display
snapshots, vmem, random state, and reset behavior with the Python reference on
both bundled fixtures. Logic tests use small ASCII circuits and the vmem fixture.
Renderer tests exercise pixel formats, indexed/RGB displays, scaling, scan
directions, and surface locks. CLI tests cover arguments and removed modes.
