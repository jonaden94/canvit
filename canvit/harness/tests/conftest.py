"""Fixtures shared by the harness tests."""

import pytest
import torch


@pytest.fixture
def one_cpu_thread():
    """Run the test with ONE torch CPU thread, then restore the previous count.

    The CPU digests (``test_task_digests.py``, ``test_rollout_parity.py``) were recorded with one
    thread: the order of a reduction's partial sums depends on the thread count, so with more threads
    every digest moves although nothing in the code changed (measured 2026-11-23, unchanged main on
    an AMD EPYC 7513 node: OMP_NUM_THREADS=1 -> 12 passed, 16 threads -> 12 failed,
    16 threads + torch.set_num_threads(1) -> 12 passed). Pinning it here makes the digests
    independent of OMP_NUM_THREADS and of the node's core count.
    """
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)
