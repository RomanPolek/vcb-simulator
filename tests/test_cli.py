import unittest
from contextlib import redirect_stderr
from io import StringIO
import main


class LeanCliTests(unittest.TestCase):
    def test_removed_experimental_modes_are_rejected(self):
        for flag in ('--smt', '--emit-c', '--pgo-optimize'):
            with self.subTest(flag=flag), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    main.parse_args(['--project', 'fixtures/test2.vcb', flag])
