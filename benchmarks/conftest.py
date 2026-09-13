"""Benchmark configuration.

JAX is configured before it is imported by any benchmark module so that the
measurements are as reproducible as possible: computations stay on the CPU
backend, and the two knobs below remove the machine the benchmark happens to
run on from the measurement. CodSpeed compares a run against the *base commit's*
run, so anything that varies with the runner shows up as a code regression that
no commit caused.

"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# TSL sizes both the XLA CPU intra-op pool and the thunk executor pool from
# `MaxParallelism()`, which reads `NPROC` and otherwise falls back to the core
# count -- so an unpinned run measures one worker's worth of dispatch per core.
# This replaces `intra_op_parallelism_threads=1`, which used to be appended to
# XLA_FLAGS below and never did anything: it is a TensorFlow ConfigProto option,
# not an XLA flag (spelled `--intra_op_parallelism_threads` XLA rejects it as
# unknown; spelled without the dashes, as it was, XLA silently drops it as a
# stray positional argument).
os.environ.setdefault("NPROC", "1")

os.environ["XLA_FLAGS"] = " ".join(
    [
        os.environ.get("XLA_FLAGS", ""),
        "--xla_cpu_multi_thread_eigen=false",
        # XLA picks its vector width from the host CPU, so the same source
        # compiles to different code on an AVX-512 runner than on an AVX2 one,
        # and the instruction count moves with it. GitHub's `ubuntu-24.04` pool
        # mixes CPU generations, which is why benchmarks flip between two stable
        # levels across runs of the same commit. Cap it at the common
        # denominator -- every x86-64 runner GitHub offers has AVX2 -- so the
        # codegen is a property of the commit. No-op on arm64.
        "--xla_cpu_max_isa=AVX2",
    ],
).strip()

# XLA only reports a token it does not recognise when that token starts with
# `--`; a bare one is silently dropped as a positional argument, which is how
# `intra_op_parallelism_threads=1` sat here doing nothing. Between this and
# XLA's own fatal "Unknown flag in XLA_FLAGS", a flag that does not take effect
# is now always loud.
if stray := [f for f in os.environ["XLA_FLAGS"].split() if not f.startswith("--")]:
    msg = f"XLA_FLAGS entries must start with `--`; XLA will ignore {stray}"
    raise ValueError(msg)
