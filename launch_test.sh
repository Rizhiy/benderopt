isort . --check &&
black . --check &&
pytest benderopt --cov-config=.coveragerc --cov=./benderopt --cov-fail-under=10 --cov-report=term-missing
