PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: help setup data tiers syllabus selftest diagnose refine pilots \
        baseline tune matrix final report figures repro clean-cache status watch

# Where has the study got to? Safe to run while run_all.sh is mid-flight.
status:
	@$(PY) src/status.py

# Live view of the stage that is decoding right now.
watch:
	@tail -f logs/$$(ls -t logs | grep -v lock | grep -v run_all.log | head -1)

help:
	@echo "Stages follow §11 of the plan; each gate must pass before the next."
	@echo "  make setup      venv + pinned dependencies"
	@echo "  make data       download, extract, cut audio, verify against published figures"
	@echo "  make diagnose   measure the segment-boundary defect (§3.4)"
	@echo "  make refine     tune + apply boundary refinement, then validate"
	@echo "  make tiers      freeze Tier 1 / 2 / 3 (§3.3)"
	@echo "  make syllabus   build chunk index, freeze the term lexicon (§5)"
	@echo "  make selftest   self-tests for scoring, correction, gating, guards"
	@echo "  make pilots     §4.2 model selection + §4.3 language configuration"
	@echo "  make baseline   B0 on Tier 1 + validation gate + headroom (§8.1, §8.4)"
	@echo "  make tune       Tier-1 sweeps: every hyperparameter chosen here (§3.3)"
	@echo "  make matrix     the full matrix on Tier 2 + stats + report (§9)"
	@echo "  make final      B0 and the best system on Tier 3 (§11 stage 11)"

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	mkdir -p data/raw/slr104
	cd data/raw/slr104 && curl -L -C - --retry 3 \
	  -o Hindi-English_test.tar.gz \
	  https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz
	cd data/raw/slr104 && tar -xzf Hindi-English_test.tar.gz
	$(PY) src/prepare_slr104.py

diagnose:
	$(PY) src/diagnose_segments.py --n 20 --offsets=-2,-1,0,1,2,3

refine:
	$(PY) src/tune_refinement.py --n 20
	$(PY) src/refine_segments.py
	$(PY) src/make_tiers.py --force
	$(PY) src/diagnose_segments.py --validate --n 20

tiers:
	$(PY) src/make_tiers.py

syllabus:
	$(PY) src/build_syllabus.py

selftest:
	$(PY) src/selftest.py

pilots:
	$(PY) src/pilots.py both --tier tier1

baseline:
	$(PY) src/run_matrix.py baseline --tier tier1

tune:
	$(PY) src/run_matrix.py tune --tier tier1

matrix:
	$(PY) src/run_matrix.py matrix --tier tier2

final:
	$(PY) src/run_matrix.py final --tier tier3 --best G

report:
	$(PY) src/bootstrap.py --tier tier2
	$(PY) src/make_report.py --tier tier2
	$(PY) src/figures.py --tier tier2

figures:
	$(PY) src/figures.py --tier tier2

repro:
	$(PY) src/repro.py

# Decodes are the expensive artefact; never delete them casually.
clean-cache:
	@echo "This deletes every cached decode. Type 'yes' to continue:"; read ans; \
	  [ "$$ans" = yes ] && rm -rf cache/asr || echo "aborted"
