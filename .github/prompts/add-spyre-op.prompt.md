---
name: add-spyre-op
description: Enable a new operation on the Spyre compiled path end-to-end
argument-hint: <aten-op-name> e.g. torch.ops.aten.hardswish or hardswish
agent: agent
tools: ['search/codebase', 'edit', 'runCommands']
---

# Add Spyre Operation

Enable the operation named in the user's input on the Spyre compiled path.

Op name: ${input:opName}

## Before writing any code

Read these files in order:

1. `.claude/skills/add-spyre-operation/SKILL.md`
2. `.claude/skills/add-spyre-operation/decision-tree.md`
3. `.claude/skills/add-spyre-operation/op-checklist.md`
4. `.claude/skills/write-spyre-op-test/SKILL.md`
5. `docs/source/compiler/adding_operations.md`

Search the codebase for similar ops and match their style exactly.

## Pattern selection

Walk `decision-tree.md` and state your chosen pattern (1, 2, or 3) with
rationale before making any edits.

| Pattern | Files |
| --- | --- |
| 1 Direct OpFunc | `torch_spyre/_inductor/spyre_kernel.py` |
| 2 Decomposition | `torch_spyre/_inductor/decompositions.py` |
| 3 Custom op | `customops.py` + `lowering.py` + `spyre_kernel.py` |
| SuperDSC (if needed) | `codegen/compute_ops.py` or `codegen/data_ops.py` |
| Constants | `torch_spyre/_inductor/constants.py` |
| Fallback | `torch_spyre/fallbacks.py` |

## Implement

Follow `op-checklist.md`. Match naming and structure of the nearest existing
op in the same file.

If the op introduces a new SuperDSC operation type, also update
`torch_spyre/_inductor/codegen/superdsc.py` dispatch.

## Model verification (required before unit tests)

Run:

```bash
pytest -c pytest_models.ini -s -rsapd tests/models/test_model_ops.py -k torch_<op>
```

Convert dots to underscores in the `-k` filter (`torch.add` → `torch_add`).

Do not proceed to unit tests if model verification reveals a new regression.

## Unit tests + OOT config

1. Add a `PARAMS` entry to `tests/inductor/test_inductor_ops.py` using
   `ParameterizedTestMeta` and `compare_with_cpu` (or the compare helper
   used by the nearest similar op).
   Do **not** hardcode one fixed 1D/2D/3D/4D shape set. Search
   `tests/inductor/test_inductor_ops.py` for the nearest existing op of
   the same kind (pointwise, binary, scalar, reduction, matmul, etc.) and
   reuse that entry's `param_sets`, dtypes, and input helpers
   (`cached_randn`, `unique_randn_along_dim`, `abs=True`, etc.).
2. Update the matching shard in `tests/configs/torch_spyre_tests/inductor/`
   — extend the `names` regex to cover new `TestOps::test_<prefix>.*` methods
   with `mode: mandatory_success`.
3. Optionally add an eager test in `tests/test_ops.py` if the op has eager
   support.

## Verify before done

```bash
python3 -m pytest tests/inductor/test_inductor_ops.py -k "<op>" -v
make check-all-configs TEST_FILE=tests/inductor/test_inductor_ops.py
make precommit
```

## Non-negotiable conventions

- Apache 2.0 license header on every new file
- `import regex as re`, never `import re`
- Google Python/C++ style; 88-char line length
- Prefer `torch.float16` and stick-aligned cases when the nearest similar
  op does; otherwise match that op's dtype/shape pattern
- `torch.manual_seed(0xAFFE)` in test class
- Match `compare_with_cpu` / tolerances used by the nearest similar test
- `WrapperHandler` wrap for `inner_fn` changes — never reconstruct `inner_fn`
- `git commit -s` for DCO; do not push unless asked

## Final output

Summarize: pattern chosen, files changed, test names added, verification
results, and any deferred work (docs, eager tests, fallbacks).
