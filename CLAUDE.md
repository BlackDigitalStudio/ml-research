# CLAUDE.md — operating frame for this repo (read first)

## The asset is information, and there are two distinct modes

The asset of this project is **information**: under what conditions a given
approach produces harvestable alpha. Two epistemic modes exist. Do not
confuse them, and do not let the second silently replace the first:

- **Exploratory conditional-optimization** — the DEFAULT for
  model-architecture research (the HD1 / `*-seq` tiers, and any "sweep this
  model" work). The question is: *under what conditions (symbol, horizon,
  context length, objective, head, data scale, …) does THIS model yield the
  most alpha, and how large / how seed-stable is it?* The deliverable is a
  **response surface**, not a verdict.
- **Confirmatory economic deploy-gate** — HM1 §5 (`research/`-frozen:
  `hd1_seq_core.py` `gate_cell` / `status_for_cell` / `BASELINE_REF_RIC`).
  The question is: *is an effect large/robust enough vs the incumbent
  baseline to risk capital?* This is a **separate, secondary** question.

## The recurring failure to avoid (this is why this file exists)

Every agent that touches this research drifts toward collapsing the
exploratory map into the confirmatory gate — leading with *"axis inert / no
real effect / refuted / escalate / doesn't beat the baseline"*. **That
output is destructive**: it discards the continuous signal (the alpha
surface and the conditions that drive it), which is the actual asset.

The pull has identifiable sources, and naming them is the defense:
1. the substrate computes `delta = rank_IC − BASELINE_REF_RIC` as its
   primary quantity and hard-codes a BINDING §5 pass/fail gate +
   `refuted/suspect/confirmed` statuses — the medium produces gate-shaped
   output by construction;
2. the pre-registration ritual (fixed rule, noise floor, weak control) is
   correct for *confirmatory* testing and gets reflexively misapplied to
   *exploratory* characterization, then self-reinforced because the ledger
   is full of it;
3. the root HD1 hypothesis is itself "does TCN beat the snapshot baseline",
   re-pasted into every rev's `framing` field;
4. agents are disposed toward decisive, defensible closure (a binary
   verdict reads as rigorous and "done"; a continuous claim is more exposed).

## Rules for any model-architecture / `*-seq` tier work

1. **The primary deliverable of a tier is the conditional alpha surface**:
   `rank_IC` and its seed-sd as a function of the swept conditions, the
   regime that **maximizes** alpha per cell, its magnitude, and its
   seed-stability. Report this first, as the headline, always.
2. The §5 gate / delta-vs-baseline / `refuted`-status machinery is a
   **secondary annotation** answering the separate deploy question. Keep it
   (see rule 4) but never as the headline or the framing of a tier result.
3. **Banned as a headline, lead sentence, or rev-statement opener:**
   "inert", "no real effect", "refuted", "escalate", "beats baseline y/n",
   or any binary verdict standing in for the surface. State the surface and
   its argmax instead. The gate verdict may appear only inside a clearly
   labelled secondary field.
4. Pre-registration and the frozen §5 gate are **NOT removed**. They remain
   valid and binding *for the confirmatory deploy question only*. Do not
   destroy frozen infra to fix framing — that is the same over-correction
   disease. The fix is demotion, not deletion.
5. Each tier characterizes *that model's own* alpha conditions. "Beats
   XGBoost / the snapshot baseline" is not the tier question.

## Mechanics (unchanged, still binding)

- Develop on the assigned feature branch; commit + push; never push
  elsewhere without explicit permission.
- `research/hypotheses.jsonl` and `experiments.jsonl` are **append-only
  event logs** — never mutate a recorded rev/result; add a new one.
- Pre-register a frozen spec for a tier *before* running it; for tiers with
  a real cost gate (e.g. egress-rebuild), scope the cost and get sign-off
  before building/running.
- The model identifier from the system prompt stays out of commits, PRs,
  code comments, and any pushed artifact — chat replies only.
