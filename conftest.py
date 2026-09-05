"""Root-level pytest conftest.

`toy_lipschitz` calls `torch.set_default_dtype(torch.float64)` at import time
in every module (a deliberate, documented choice - CLAUDE.md: "so
true-vs-estimate comparisons aren't contaminated by float32 noise").
That is process-wide `torch` state, not scoped to the package.
When a single `pytest` invocation collects `toy_lipschitz`'s tests before
`mnist_lipschitz`'s or `signature_distance`'s, the float64 default leaks into
every test collected afterward and silently changes tensor dtypes there,
surfacing as `RuntimeError: Float did not match Double` in code that assumes
float32 (confirmed: `signature_distance`'s tests pass 129/129 in isolation but
fail when run after `toy_lipschitz` in the same process).

This fixture resets the default dtype before every test, based on which
package the test belongs to, so a combined invocation
(`pytest toy_lipschitz/tests mnist_lipschitz/tests signature_distance/tests`)
behaves identically to running each package separately. It changes no
experiment logic - only test-time global state that is otherwise
order-dependent.
"""

import torch
import pytest


@pytest.fixture(autouse=True)
def _reset_torch_default_dtype(request):
    module_name = request.module.__name__ if request.module else ""
    if module_name.startswith("toy_lipschitz"):
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(torch.float32)
