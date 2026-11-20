"""CPU tests for distill's WebDataset multi-job (SLURM-array) resume.

The shard schedule is deterministic in (job_index, world_size, batch_size_per_gpu,
steps_per_job, samples_per_shard): job j reads the slice at flat offset
``job_index * shards_per_gpu * world_size``. So resuming at the wrong job_index — or
with any of those inputs changed — silently re-processes or skips training data.
These pin the guards that make both loud instead: the start-step derivation
(``(saved job_index + 1) * steps_per_job``) and the invariant check.

What is real here: the derivation, the invariant comparison, and the checkpoint
round-trip (leg 1's ``resume_state()`` is what leg 2 reads). What is stubbed: the
loader itself (a real one needs tar shards on disk) and the normalizer init. The
same two legs against the real webdataset: ``canvit_claude/canvit/unification/harness_run_wds_resume.py``.
"""

from types import SimpleNamespace

import pytest

from canvit.ade20k.task import Ade20kRunTask
from canvit.distill.config import Config
from canvit.distill.data.webdataset import WebDatasetTrainLoader
from canvit.distill.task import DistillRunTask

_BS, _SPJ, _SPS = 8, 64, 512


def _task(**over):
    cfg = Config(webdataset_dir="/nonexistent", batch_size_per_gpu=_BS, steps_per_job=_SPJ, **over)
    t = DistillRunTask(cfg)
    t.scene_norm = SimpleNamespace(initialized=True)  # skip the first-shard normalizer init
    return t


def _stub_loaders(monkeypatch, *, samples_per_shard=_SPS):
    """Patch create_loaders to hand back a bare WebDatasetTrainLoader; record its kwargs."""
    seen: dict = {}
    loader = object.__new__(WebDatasetTrainLoader)  # no tar shards needed for these tests
    loader.samples_per_shard = samples_per_shard

    def fake(cfg, *, job_index, world_size, rank):
        seen.update(job_index=job_index, world_size=world_size)
        return SimpleNamespace(train=loader, val=None)

    monkeypatch.setattr("canvit.distill.data.create_loaders", fake)
    return seen


def _ckpt(task):
    """The checkpoint metadata run() would write for this task (see harness/run.py)."""
    return {"metadata": {"resume_state": task.resume_state()}}


def _sched(last_epoch):
    return SimpleNamespace(last_epoch=last_epoch)


# --- the two legs ---------------------------------------------------------
def test_fresh_job_reads_slice_zero_and_records_it(monkeypatch):
    seen = _stub_loaders(monkeypatch)
    t = _task()
    t.build_loaders(world_size=1, rank=0)
    assert seen["job_index"] == 0
    assert _ckpt(t)["metadata"]["resume_state"] == {
        "job_index": 0, "ddp_world_size": 1, "batch_size_per_gpu": _BS,
        "steps_per_job": _SPJ, "samples_per_shard": _SPS,
    }


def test_second_leg_starts_at_the_next_job_slice(monkeypatch):
    """Leg 1 runs job 0 and saves; leg 2 must start at step steps_per_job AND read the
    NEXT shard slice (job_index 1) — not re-read job 0's."""
    seen = _stub_loaders(monkeypatch)
    leg1 = _task()
    leg1.build_loaders(world_size=1, rank=0)
    payload = _ckpt(leg1)

    leg2 = _task()
    start_step = leg2.resume_start_step(payload, _sched(_SPJ))  # scheduler agrees: 1 job done
    assert start_step == (0 + 1) * _SPJ
    leg2.build_loaders(world_size=1, rank=0)
    assert seen["job_index"] == 1, "leg 2 must consume the shard slice after leg 1's"
    # `start_step` is no longer passed to create_loaders (it only ever positioned the
    # deleted sharded loader's cursor); the resume step itself is asserted above and is
    # consumed by harness/run.py, not by the loader.
    assert leg2.resume_state()["job_index"] == 1  # leg 3 would then get 2


def test_third_leg_keeps_advancing(monkeypatch):
    _stub_loaders(monkeypatch)
    t = _task()
    assert t.resume_start_step({"metadata": {"resume_state": {"job_index": 6}}},
                               _sched(7 * _SPJ)) == 7 * _SPJ


# --- refusals (each would otherwise silently corrupt the data schedule) ----
def test_checkpoint_without_job_index_raises():
    t = _task()
    with pytest.raises(RuntimeError, match="job_index"):
        t.resume_start_step({"metadata": {"resume_state": {}}}, _sched(_SPJ))
    with pytest.raises(RuntimeError, match="job_index"):
        t.resume_start_step({}, _sched(_SPJ))  # e.g. a pre-resume_state checkpoint


def test_midjob_save_raises_instead_of_offsetting_the_schedule():
    """A SIGUSR1 save at step 40 of a 64-step job: resuming would jump the LR schedule
    to 64 and skip the rest of job 0's shards."""
    t = _task()
    with pytest.raises(RuntimeError, match="end-of-job boundary"):
        t.resume_start_step({"metadata": {"resume_state": {"job_index": 0}}}, _sched(40))


@pytest.mark.parametrize(
    ("key", "bad"),
    [("ddp_world_size", 2), ("batch_size_per_gpu", 16), ("steps_per_job", 128),
     ("samples_per_shard", 1024)],
)
def test_changed_schedule_input_raises(monkeypatch, key, bad):
    saved = {"job_index": 0, "ddp_world_size": 1, "batch_size_per_gpu": _BS,
             "steps_per_job": _SPJ, "samples_per_shard": _SPS}
    saved[key] = bad  # the checkpoint was written under a DIFFERENT schedule
    _stub_loaders(monkeypatch)
    t = _task()
    t.resume_start_step({"metadata": {"resume_state": saved}}, _sched(_SPJ))
    with pytest.raises(RuntimeError, match="shard-schedule offset would be wrong"):
        t.build_loaders(world_size=1, rank=0)


def test_matching_schedule_inputs_pass(monkeypatch):
    seen = _stub_loaders(monkeypatch)
    t = _task()
    t.resume_start_step({"metadata": {"resume_state": {
        "job_index": 0, "ddp_world_size": 1, "batch_size_per_gpu": _BS,
        "steps_per_job": _SPJ, "samples_per_shard": _SPS}}}, _sched(_SPJ))
    t.build_loaders(world_size=1, rank=0)
    assert seen["job_index"] == 1


# --- ade20k is the map-style exception (no shard schedule) ----------------
# in1k now uses THIS shard schedule too (its resume round-trip + invariant guards are
# covered in tasks/tests/test_run_wrappers.py::test_in1k_shard_schedule_*).
def test_ade20k_carries_no_schedule_state():
    from canvit.ade20k.config import Ade20kConfig

    t = Ade20kRunTask(Ade20kConfig(tracker="none"))
    assert t.resume_state() == {}
    assert t.resume_start_step({}, _sched(7)) == 7
