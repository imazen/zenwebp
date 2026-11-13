"""Exercise the full runner with Cargo's old and new artifact layouts."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TEST = "decoder::alloc_util::tests::valid_layout_allocation_failure"
RUNNER = Path(__file__).with_name("run-built-i686-allocation-test.py")


class ArtifactPaths(unittest.TestCase):
    def check_layout(self, relative, valid=True):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            binary = root / relative
            binary.parent.mkdir(parents=True)
            header = bytearray(20)
            header[:6] = b"\x7fELF\x01\x01" if valid else b"\x7fELF\x02\x01"
            header[18:20] = (3).to_bytes(2, "little")
            binary.write_bytes(header)
            receipt = root / "artifacts.json"
            receipt.write_text(json.dumps({"reason": "compiler-artifact",
                "target": {"name": "zenwebp"}, "profile": {"test": True},
                "executable": str(Path("/target") / relative)}) + "\n")
            with patch.object(sys, "argv", [str(RUNNER), str(receipt), str(root)]), patch(
                "subprocess.run", return_value=subprocess.CompletedProcess([], 0, f"{TEST}: test\n")
            ) as run:
                if valid:
                    runpy.run_path(str(RUNNER), run_name="__main__")
                    self.assertEqual(run.call_count, 2)
                    self.assertEqual(run.call_args.args[0][0], str(binary.resolve()))
                else:
                    with self.assertRaisesRegex(RuntimeError, "not a 32-bit"):
                        runpy.run_path(str(RUNNER), run_name="__main__")
                    run.assert_not_called()

    def test_deps_layout(self):
        self.check_layout("i686-unknown-linux-gnu/debug/deps/zenwebp-abc")

    def test_build_out_layout(self):
        self.check_layout("i686-unknown-linux-gnu/debug/build/zenwebp/hash/out/zenwebp-abc")

    def test_wrong_elf_is_refused(self):
        self.check_layout("i686-unknown-linux-gnu/debug/deps/zenwebp-abc", valid=False)


if __name__ == "__main__":
    unittest.main()
