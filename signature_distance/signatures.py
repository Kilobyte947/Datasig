"""Truncated signature computation, shared by both methods' 2D
`[t, value]` streams (Method A's `patch_sv_stream`, Method B's
`line_stream`) - see PLAN.md Phase 2.

Built on roughpy_jax's low-level, JAX-native primitives (`Lie`, `cbh`,
`to_signature`) rather than its higher-level `Stream` object wrappers
(`LieIncrementStream` / `PiecewiseAbelianStream`): those primitives are
`custom_vjp`-registered JAX functions, so `jax.vmap` batches this cleanly
(verified below), and their behavior is fully pinned down by two closed-form
checks (a straight line's exact tensor-exponential signature, and an
L-shaped path's hand-computed area term) rather than inferred from sparse
docs on the Stream classes.

Each stream is piecewise-linear by construction (straight-line interpolation
between consecutive sampled points, per PLAN.md's Method A/B design). For a
single straight segment, the log-signature at any truncation depth is
*exactly* its degree-1 displacement vector (a straight line has no
higher-order/area terms) - so each increment is a pure "abelian" Lie element,
and `cbh` (Campbell-Baker-Hausdorff) combines the whole sequence of segments
into the log-signature of the complete path, matching Chen's identity.
"""

import numpy as np
import roughpy_jax as rpj
import torch
from jax import numpy as jnp
from jax import vmap

WIDTH = 2  # every stream in this project is 2D: [t, value]


def _lie_basis(depth: int):
    lie_basis = rpj.LieBasis(width=WIDTH, depth=depth)
    tensor_basis = rpj.to_tensor_basis(lie_basis)
    return lie_basis, tensor_basis


def signature_of_stream(stream: torch.Tensor, depth: int) -> torch.Tensor:
    """Truncated signature of a batch of 2D piecewise-linear streams.

    stream: (batch, K, 2) float, a piecewise-linear path per batch element
            (columns [t, value], consecutive points joined by straight
            lines - the convention used throughout this project).
    depth: truncation depth (signature dimension is 1+2+4+...+2**depth).
    returns: (batch, 1 + 2 + 4 + ... + 2**depth) float32 - the truncated
             signature per batch element, index 0 is always 1.0 (the
             constant term), indices [1:3] are the net displacement
             (stream[:, -1] - stream[:, 0]).
    """
    lie_basis, tensor_basis = _lie_basis(depth)
    basis_size = lie_basis.size()
    pad_len = basis_size - WIDTH

    stream_jax = jnp.asarray(stream.detach().cpu().numpy(), dtype=jnp.float32)

    def signature_of_one(points):
        increments = points[1:] - points[:-1]  # (K-1, 2)
        padded = jnp.concatenate(
            [increments, jnp.zeros((increments.shape[0], pad_len), dtype=jnp.float32)],
            axis=1,
        )  # (K-1, basis_size)
        lie_pieces = [rpj.Lie(padded[i], lie_basis) for i in range(padded.shape[0])]
        log_sig = rpj.cbh(*lie_pieces, lie_basis=lie_basis)
        sig = rpj.to_signature(log_sig, tensor_basis)
        return sig.data

    sig_batch = vmap(signature_of_one)(stream_jax)
    return torch.from_numpy(np.array(sig_batch)).to(torch.float32)
