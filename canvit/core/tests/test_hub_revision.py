"""The Hub revision pin for canvit/* repos (see canvit.core.checkpoints.hub_revision).

Upstream rewrote `main` of the published repos in a format this code cannot read
(2026-09-26); unpinned, the flagship fails to construct and probes could load with
renamed keys left at random init behind only a warning.
"""

from canvit.core.checkpoints import LEGACY_HUB_REVISION, LEGACY_POLICY_HUB_REVISION, hub_revision


def test_converted_canvit_repos_pin_the_legacy_tag():
    for name in ["canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02",
                 "canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06",
                 "probe-ade20k-40k-s512-c64-in21k"]:
        assert hub_revision(f"canvit/{name}") == LEGACY_HUB_REVISION


def test_policy_scorers_pin_their_own_tag():
    assert hub_revision("canvit/qpolicy-ade20k-c64-t5-qband-2026-07-04-s2") == LEGACY_POLICY_HUB_REVISION
    assert hub_revision("canvit/pgpolicy-ade20k-c64-t5-2026-07-05-s0") == LEGACY_POLICY_HUB_REVISION


def test_unchanged_and_non_hub_sources_stay_on_main(tmp_path):
    assert hub_revision("canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe") is None
    assert hub_revision("facebook/dinov3-vitb16-pretrain-lvd1689m") is None
    local = tmp_path / "canvit" / "probe-x"
    local.mkdir(parents=True)
    assert hub_revision(str(local)) is None          # absolute local dir
    assert hub_revision("canvit") is None            # no repo name
