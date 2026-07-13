---
# ADR: ops

**Date:** 2026-07-13
**Status:** Active
**Author:** ask-z-doc-bot
**Commit:** local

## Context
This ADR documents architectural changes to TorchSpyre's ops module that introduce in-place tensor filling operations leveraging a new device-side FillDMA kernel. The existing implementation used CPU-side temporary tensors for fill operations, which incurred host-device transfer overhead. The goal is to improve performance for common tensor initialization patterns while maintaining API compatibility.

## Decision
Refactor `spyre_full`, `spyre_ones`, and `spyre__zero_` to directly call `torch_spyre._C.fill_tensor` with appropriate fill values, eliminating CPU-side temporary tensors. Added type and value validation for fill parameters (disallowing complex numbers) and layout/device restrictions. For `spyre__zero_`, replaced the previous CPU-based zeroing approach with the efficient device-side FillDMA kernel.

## Consequences
**Positive:**
- Reduced memory bandwidth usage by eliminating unnecessary host-device data movement
- Faster tensor initialization for large tensors, especially on GPUs
- Cleaner API surface by exposing dedicated fill operations

**Negative:**
- Slight increase in code complexity within the ops module
- Potential for subtle bugs if existing code relies on side effects of the old implementation
- Limited to supported dtypes and layouts (no complex fill values)

## Implementation Notes
- Use `torch_spyre._C.fill_tensor(self, value)` for in-place filling, converting `fill_value` to float when necessary
- Validate `layout` against `torch.strided` and `None` explicitly
- Disallow `pin_memory=True` for these operations as DMA handling differs
- Add clear error messages for unsupported types (complex) and parameters (pin_memory)
- Update documentation to reflect new in-place fill semantics and restrictions
---
