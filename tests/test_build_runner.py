"""Build and state-equivalence checks for the production native event engine."""
import ctypes
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from runner.native import NativeRuntime
from runner.simulation import CircuitRuntime


class BuildRunnerTests(unittest.TestCase):
    def test_project_is_required(self):
        with patch('sys.stderr', new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                main.parse_args([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('required: --project', stderr.getvalue())

    def test_explicit_project(self):
        self.assertEqual(main.parse_args(['--project', 'fixtures/gpu7.vcb']).project, Path('fixtures/gpu7.vcb'))

    def test_benchmark_rejects_invalid_durations(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    main.parse_args(['--project', 'fixtures/test2.vcb', '--benchmark', value])

    def test_native_matches_reference_and_reset(self):
        for fixture in ('test2.vcb', 'gpu7.vcb'):
            with self.subTest(fixture=fixture):
                context = main.build_runner_context(str(Path('fixtures') / fixture))
                reference = CircuitRuntime(context)
                native = NativeRuntime(context)
                native.native.native_rng_state.restype = ctypes.POINTER(ctypes.c_uint32)
                for ticks in (0, 1, 2, 7, 53, 256):
                    reference.tick(ticks)
                    native.tick(ticks)
                    self.assertEqual(native.make_snapshot(), reference.make_snapshot())
                    self.assertEqual(native._memory.tolist(), list(reference.vmem_words))
                    self.assertEqual(list(native.native.native_rng_state()[:reference.gate_count + 1]), reference.rng_state)
                native.reset()
                reference.reset()
                self.assertEqual(native.make_snapshot(), reference.make_snapshot())
                native.tick(31)
                reference.tick(31)
                self.assertEqual(native.make_snapshot(), reference.make_snapshot())

    def test_timed_native_batch_matches_exact_tick_reference(self):
        context = main.build_runner_context('fixtures/gpu7.vcb')
        native, reference = NativeRuntime(context), CircuitRuntime(context)
        native.native.native_rng_state.restype = ctypes.POINTER(ctypes.c_uint32)
        native.tick_for(0)
        self.assertEqual(native.tick_count, 0)
        for _ in range(3):
            start = native.tick_count
            native.tick_for(0.0002)
            count = native.tick_count - start
            self.assertGreater(count, 0)
            reference.tick(count)
            self.assertEqual(native.make_snapshot(), reference.make_snapshot())
            self.assertEqual(native._memory.tolist(), list(reference.vmem_words))
            self.assertEqual(list(native.native.native_rng_state()[:reference.gate_count + 1]), reference.rng_state)

    def test_instances_do_not_share_state(self):
        context = main.build_runner_context('fixtures/test2.vcb')
        first, second = NativeRuntime(context), NativeRuntime(context)
        initial = second.make_snapshot()
        first.tick(100)
        self.assertEqual(second.make_snapshot(), initial)

    def test_missing_compiler_error(self):
        context = main.build_runner_context('fixtures/test2.vcb')
        with self.assertRaisesRegex(RuntimeError, 'C compiler not found'):
            NativeRuntime(context, compiler='vcb-missing-compiler')


if __name__ == '__main__':
    unittest.main()
