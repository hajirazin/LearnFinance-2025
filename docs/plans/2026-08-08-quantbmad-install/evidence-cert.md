# Evidence cert
plan_path: docs/plans/2026-08-08-quantbmad-install/plan.md
hypothesis: Installing QuantBMAD per docs/quantbmad-spec-v3.md makes Research-glob changes route through qb-plan/qb-implement with CI backstop, without reimplementing BMM.
experiment_path: _bmad/quantbmad/tests/
command: python3 -m pytest _bmad/quantbmad/tests -q && python3 _bmad/quantbmad/scripts/sync_skill_mirrors.py --check && python3 _bmad/quantbmad/scripts/check_research_gate.py --mirrors-only
exit_code: 0
key_metrics: unit tests pass; skill mirror checksums match; no symlinks
falsification_criteria_checked: yes — globs/gate/ledger covered by unit tests; mirrors verified
author_model: cursor-agent-implementer
skeptic_model: cursor-agent-install-verification
skeptic_task_id_or_fresh_chat: fresh-chat:2026-08-08-quantbmad-install
verdict: PASS
written_by: qb-agent-skeptic
