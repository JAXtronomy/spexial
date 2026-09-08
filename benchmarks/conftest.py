"""Benchmark configuration.

JAX is configured before it is imported by any benchmark module so that the
measurements are as reproducible as possible: computations stay on the CPU
backend and the XLA host thread pool is limited to a single thread, which keeps
the amount of work attributed to a benchmark independent of the number of cores
of the machine running it.

"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = " ".join(
    [
        os.environ.get("XLA_FLAGS", ""),
        "--xla_cpu_multi_thread_eigen=false",
        "intra_op_parallelism_threads=1",
    ],
).strip()
