# Party consensus
kind: plan | implement
plan_or_impl: <path to plan.md>
# kind:implement — ship consensus AFTER code; packet should include git_diff.
# kind:plan — consensus on a draft plan (opt-in party; no production Research-glob edits yet).
git_diff: <required when kind:implement — summary path or inline basis, e.g. docs/plans/<slug>/implement.diff>
round: 1
members: q-researcher, q-pm, q-ml, q-risk, q-skeptic, q-validation, q-dev, q-architect
votes:
  q-researcher: agree | disagree — <one-line reason>
  q-pm: agree | disagree — <one-line reason>
  q-ml: agree | disagree — <one-line reason>
  q-risk: agree | disagree — <one-line reason>
  q-skeptic: agree | disagree — <one-line reason>
  q-validation: agree | disagree — <one-line reason>
  q-dev: agree | disagree — <one-line reason>
  q-architect: agree | disagree — <one-line reason>
unanimous: true | false
outcome: approve | escalate-to-razin
razin_decision: <filled only if escalated — verbatim human decision>
