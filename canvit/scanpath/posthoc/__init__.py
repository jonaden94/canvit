"""Post-hoc bias corrections of a finished policy's fixation map (``priors``: temperature and
amplitude priors; ``saccade_cost``: the per-checkpoint saccade-length fit).

Post hoc = fitted on top of a finished policy's map; a diagnostic, not training. It is kept
apart from the evaluation tool (``canvit.scanpath``: data, models, evaluate) and from
training: the training-side saccade cost lives in the harness (branch ``saccade-cost``,
``JointPolicyConfig.saccade_cost*``).

Written by Claude (AI, Claude Opus 5.5), 2026-11-25, at Valentin's request."""
