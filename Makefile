PY=.venv/bin/python
setup:
	/opt/homebrew/bin/python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
test:
	$(PY) -m pytest -q tests
reproduce:
	./scripts/run_all.sh
