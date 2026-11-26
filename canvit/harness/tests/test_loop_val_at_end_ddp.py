"""Validate-at-end under DDP (2 CPU processes, gloo): every rank must take the same branch.

Each job validates its final weights; the next job skips its start validation when
``last_val.json`` (written by rank 0) says that step is validated. If the ranks decided
differently, one would enter the validation's collective while the other enters the next
one (a barrier or the metric all-reduce) and the job would hang. The validation here does
an all-reduce, like distill's distributed eval, so a disagreement shows up as a timeout.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world_size, init_file, ckpt_dir, q):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        torch.set_num_threads(1)
        from canvit.harness.loop import run_training_loop
        from canvit.harness.tests.test_loop_e2e import _batches, _probe_setup, _resume_kw

        evals: list[int] = []

        def on_eval(step):
            t = torch.tensor([float(step)])
            dist.all_reduce(t)  # a collective inside validation, as in distill's split eval
            assert t.item() == 2 * step, "ranks validated different steps"
            evals.append(step)
            return {"miou_final": 0.1 * step} if rank == 0 else {}

        kw = dict(_probe_setup(), ckpt_dir=ckpt_dir, eval_every=1, best_metric="miou_final",
                  on_eval=on_eval, is_dist=True, rank=rank)
        run_training_loop(**kw, train_batches=_batches(), n_steps=2)   # job 1: 0, 1 + end 2
        dist.barrier()  # job boundary: rank 0 has written last_val.json
        run_training_loop(**kw, train_batches=_batches(), n_steps=2, **_resume_kw(ckpt_dir))  # 3 + end 4
        q.put((rank, evals))
    except Exception as e:  # noqa: BLE001 - reported to the parent
        q.put((rank, f"raised: {e!r}"))
    finally:
        dist.destroy_process_group()


def test_ranks_agree_on_the_skipped_start_validation(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    procs = [ctx.Process(target=_worker, args=(r, 2, str(tmp_path / "pg_init"), ckpt, q)) for r in range(2)]
    for p in procs:
        p.start()
    try:
        res = dict(q.get(timeout=300) for _ in procs)  # a rank disagreement hangs -> Empty here
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
    assert res == {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3, 4]}, res


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
