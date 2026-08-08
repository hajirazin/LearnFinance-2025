# Evidence cert
plan_path: docs/plans/<slug>/plan.md
hypothesis: <text>
experiment_path: <path>
command: <exact command run>
exit_code: <int>
key_metrics: <bullet or short summary>
falsification_criteria_checked: <yes + notes>
author_model: <provider/model that authored the experiment>
skeptic_model: <provider/model that ran Skeptic — prefer different vendor>
skeptic_task_id_or_fresh_chat: <id or "fresh-chat:YYYY-MM-DD">
verdict: PASS | FAIL | AMBIGUOUS
written_by: qb-agent-skeptic
# Parent agents must not edit the verdict or written_by fields.
# Only after party consensus Agree or Razin override (3-loop escalate).
