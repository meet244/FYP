#!/usr/bin/env bash
# Unattended execution of the whole study, in the order §9.2 and §11 require.
#
# The plan budgets "on the order of twenty hours, spread across a few unattended
# overnight sessions" (§9.2). This script is that session. It is safe to interrupt and
# re-run: every decode is cached by (utterance, model, decode-config, audio version), so
# a restart resumes instead of repeating.
#
#   ./run_all.sh            run every stage in order
#   ./run_all.sh tune       run one stage and everything after it
#
# Logs land in logs/<stage>.log; progress is greppable with:
#   grep -E "WER=|DECISION|PASS|FAIL" logs/*.log

set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONPATH=src
mkdir -p logs

# Only one chain at a time. Two concurrent chains would decode the same utterances
# twice, contend for the same cores, and can race on the same cache file.
LOCK=logs/run_all.lock
if ! mkdir "${LOCK}" 2>/dev/null; then
  owner=$(cat "${LOCK}/pid" 2>/dev/null || echo "?")
  if kill -0 "${owner}" 2>/dev/null; then
    echo "run_all.sh is already running (pid ${owner}). Watch it with:"
    echo "    PYTHONPATH=src .venv/bin/python src/status.py"
    exit 1
  fi
  echo "removing stale lock from pid ${owner}"
  rm -rf "${LOCK}"; mkdir "${LOCK}"
fi
echo $$ > "${LOCK}/pid"
trap 'rm -rf "${LOCK}"' EXIT

run () {                      # run <stage-name> <command...>
  local name=$1; shift
  local log="logs/${name}.log"
  echo "=== $(date '+%F %T')  START  ${name}" | tee -a logs/run_all.log
  if "$@" > "${log}" 2>&1; then
    echo "=== $(date '+%F %T')  DONE   ${name}" | tee -a logs/run_all.log
    grep -E "WER=|DECISION|PASS|FAIL|chosen|top-1|headroom" "${log}" | tail -12
  else
    local rc=$?
    echo "=== $(date '+%F %T')  FAILED ${name} (exit ${rc}) — see ${log}" \
      | tee -a logs/run_all.log
    tail -25 "${log}"
    exit ${rc}
  fi
}

STAGES=(selftest pilots config baseline tune matrix setup)
FROM=${1:-selftest}
started=0

for stage in "${STAGES[@]}"; do
  [[ "${stage}" == "${FROM}" ]] && started=1
  [[ ${started} -eq 1 ]] || continue
  case ${stage} in
    selftest)
      # Cheap and catches scoring/pipeline regressions before hours of decoding.
      run selftest_units    ${PY} src/selftest.py
      run selftest_pipeline ${PY} src/selftest_pipeline.py
      ;;
    pilots)
      # §4.3 first: the language setting is shared by every later decode, and the
      # chosen arm doubles as the Tier-1 baseline decode.
      run pilot_language ${PY} src/pilots.py language --tier tier1
      run pilot_model    ${PY} src/pilots.py model    --tier tier1
      ;;
    config)
      run apply_pilots ${PY} src/apply_pilot_decisions.py
      ;;
    baseline)
      # §11 stages 3-4: baseline locked, validation gate inspected, headroom stated
      # before any grounded condition is run.
      run baseline_tier1 ${PY} src/run_matrix.py baseline --tier tier1
      ;;
    tune)
      # §3.3: every hyperparameter is selected here and only here.
      run tune_tier1 ${PY} src/run_matrix.py tune --tier tier1
      ;;
    matrix)
      # §9.1 on the reporting tier, then statistics, tables and figures.
      run matrix_tier2 ${PY} src/run_matrix.py matrix --tier tier2
      ;;
    setup)
      run setup_section ${PY} src/make_setup_section.py
      run checklist     ${PY} src/repro.py
      ;;
  esac
done

echo "=== $(date '+%F %T')  ALL STAGES COMPLETE" | tee -a logs/run_all.log
echo
echo "Results:   report/results_tier2.md"
echo "Setup:     report/01_dataset_and_harness.md"
echo "Figures:   report/figures/"
echo "Checklist: report/checklist.json"
echo
echo "Tier 3 (final confirmation, ~8 h per system) is deliberately separate:"
echo "  ${PY} src/run_matrix.py final --tier tier3 --best <best system from tier2>"
