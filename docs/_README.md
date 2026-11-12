# docs

Extended documentation linked from the top-level [README](../README.md). The
README describes the repository; these documents describe *procedures* — specific
training campaigns, how to launch them, and how to judge their results.

| document | contents |
|---|---|
| [`valentin.md`](valentin.md) | A short orientation for a new team member (written for Valentin, 2026-11-12): what CanViT is, the two entry points, presets and flags, what cluster jobs pin, the layout, and where the visual-search task stands. |
| [`q_policy_foveated.md`](q_policy_foveated.md) | Training the Q viewpoint policy for a **foveated** model: the two halves a policy run needs, the three settings that fail silently if mismatched, and the ready-made launcher. Run end to end once (exp36, ten seeds, 2026-08-31) and it beats random viewpoints — a first result on a foveated backbone, not a reference number. |
| [`verification_runs.md`](verification_runs.md) | The exp32–exp35 campaign: four groups of runs that verify pretraining, ImageNet-1k finetuning, ADE20K probing and viewpoint-policy training at full scale against earlier reference results. |

## Assets

| file | produced by |
|---|---|
| `assets/ComparisonPoliciesADE20K.png` | `scripts/plot_policy_comparison.py` — the learned Viewpoint-Q policy against the open-loop baselines over t = 0..4, with the paper's Table 4 rows as dashed references |
| `assets/_policy_comparison_data.json` | the measure stage of that same script (cached evaluations, so re-styling the figure costs no GPU) |
