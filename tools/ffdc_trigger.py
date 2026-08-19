#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FFDC trigger — exercises kernel_runner.py exception paths on real hardware.

Drives SpyreSDSCKernelRunner and SpyreUnimplementedRunner through their
real exception paths so FFDC fires on a genuine traceback with real
hardware state and compiler artifacts.

Run from repo root with:
    TORCH_SPYRE_FFDC=1 TORCH_COMPILE_DEBUG=1 python3 tools/ffdc_trigger.py
"""

import glob
import json
import os
import tempfile
from typing import Any

import torch
import torch_spyre

from torch_spyre.execution import kernel_runner as kr
from torch_spyre.execution.kernel_runner import (
    SpyreSDSCKernelRunner,
    SpyreUnimplementedRunner,
)
from torch_spyre.profiler._ffdc import _default_output_dir

FFDC_OUT = _default_output_dir()


def _category_pattern(category: str) -> str:
    return str(FFDC_OUT / f"ffdc_{category}_*.json")


def _snapshot_reports(pattern: str) -> set[str]:
    return set(glob.glob(pattern))


def _new_report_path(pattern: str, before: set[str]) -> str | None:
    """Return a report created after ``before``, ignoring mtime resolution."""
    new = [path for path in glob.glob(pattern) if path not in before]
    if not new:
        return None
    # Same-category filenames sort by the embedded timestamp, not st_mtime.
    return max(new)


def _print_collector_stats(collector: dict[str, Any]) -> None:
    print(
        f"  completeness={collector['completeness_pct']}%  "
        f"latency={collector['capture_latency_ms']}ms  "
        f"missing={collector['missing_fields']}"
    )


def _write_minimal_spyrecode(parent: str) -> str:
    """Write a spyreCodeDir that prepare_kernel can load; return ``parent``."""
    spyrecode_dir = os.path.join(parent, "spyreCodeDir")
    os.makedirs(spyrecode_dir, exist_ok=True)
    spyrecode_json = {
        "JobPreparationPlan": [
            {"command": "Allocate", "properties": {"size": "1024"}},
            {
                "command": "InitTransfer",
                "properties": {
                    "init_bin_file": "init_binary.bin",
                    "dev_ptr": "120259084288",
                    "size": "1024",
                },
            },
        ],
        "JobExecPlan": [
            {
                "command": "ComputeOnDevice",
                "properties": {"job_bin_ptr": "120259084288"},
            }
        ],
    }
    with open(os.path.join(spyrecode_dir, "spyrecode.json"), "w") as f:
        json.dump(spyrecode_json, f)
    with open(os.path.join(spyrecode_dir, "init_binary.bin"), "wb") as f:
        f.write(b"\x00" * 1024)
    return parent


def _get_diagnostic_report(output_dir=None):
    getter = getattr(getattr(torch, "spyre", None), "get_diagnostic_report", None)
    if getter is None:
        getter = torch_spyre.profiler.get_diagnostic_report
    if output_dir is None:
        return getter()
    return getter(output_dir=output_dir)


def _print_retrieved(output_dir=None) -> None:
    report = _get_diagnostic_report(output_dir)
    if report is None:
        print("  get_diagnostic_report: None")
        return
    print(f"  failure.category : {report['failure']['category']}")
    print(f"  _report_path     : {report['_report_path']}")


def _record_report(reports, category: str, before: set[str]) -> None:
    report_path = _new_report_path(_category_pattern(category), before)
    if report_path:
        with open(report_path) as f:
            report = json.load(f)
        reports.append((category, report))
        print(f"  Report written: {report_path}")
        _print_collector_stats(report["collector"])
        _print_retrieved()
    else:
        print("  [WARN] No report found — check FFDC output_dir")


def main():
    print("\n=== FFDC Real Trigger ===\n")
    reports = []
    os.environ.setdefault("TORCH_SPYRE_FFDC", "1")

    # ── Scenario A: runtime_launch failure ──────────────────────────────────────
    # SpyreSDSCKernelRunner.__init__ calls prepare_kernel(code_dir/spyreCodeDir).
    # Survive that with a minimal spyrecode tree, then fail in run() so
    # @with_ffdc(CATEGORY_RUNTIME_LAUNCH) fires. Do not claim runtime_launch
    # if prepare_kernel raises in __init__.
    os.environ.pop("DUMP_SPYRE_CODE", None)

    print("Scenario A: SpyreSDSCKernelRunner.run() → launch_jobplan raises")
    with tempfile.TemporaryDirectory(prefix="ffdc_spyrecode_") as tmp:
        code_dir = _write_minimal_spyrecode(tmp)
        runner = None
        try:
            torch.zeros(1, device="spyre")
            runner = SpyreSDSCKernelRunner(
                name="test_kernel_add",
                code_dir=code_dir,
            )
        except Exception as e:
            print(f"  [SKIP] prepare_kernel failed in __init__: {e}")
            print("  Not claiming runtime_launch (hook is on run(), not __init__).")

        if runner is not None:
            orig_launch = kr.launch_jobplan

            def boom(*_args, **_kwargs):
                raise RuntimeError("ffdc launch boom")

            kr.launch_jobplan = boom
            before = _snapshot_reports(_category_pattern("runtime_launch"))
            try:
                runner.run()
            except RuntimeError as e:
                print(f"  Exception re-raised (expected): {e}")
            else:
                raise AssertionError(
                    "Expected RuntimeError from runner.run() but none was raised"
                )
            finally:
                kr.launch_jobplan = orig_launch
            _record_report(reports, "runtime_launch", before)

    # ── Scenario B: unimplemented op failure ────────────────────────────────────
    print(
        "\nScenario B: SpyreUnimplementedRunner.run() → unimplemented op → FFDC fires"
    )
    urunner = SpyreUnimplementedRunner(
        name="test_kernel_fft",
        op="aten::fft_fft",
    )
    before = _snapshot_reports(_category_pattern("unimplemented"))
    try:
        urunner.run()
    except RuntimeError as e:
        print(f"  Exception re-raised (expected): {e}")
    else:
        raise AssertionError(
            "Expected RuntimeError from urunner.run() but none was raised"
        )

    _record_report(reports, "unimplemented", before)

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n=== Captured Report Fields ===")
    for cat, r in reports:
        print(f"\n[{cat}]")
        print(f"  failure.exception_type : {r['failure']['exception_type']}")
        print(f"  failure.message        : {r['failure']['message'][:80]}")
        tb = r["failure"]["traceback"]
        tb_str = tb if isinstance(tb, str) else "".join(tb)
        print(f"  failure.traceback_lines: {len(tb_str.splitlines())}")
        print(f"  metadata.torch_version : {r['metadata'].get('torch_version', 'N/A')}")
        print(
            f"  metadata.torch_spyre_version : "
            f"{r['metadata'].get('torch_spyre_version', 'N/A')}"
        )
        print(f"  artifacts.found_count  : {r['artifacts']['found_count']}")
        print(f"  hardware_state         : {r['hardware_state']}")


if __name__ == "__main__":
    main()
