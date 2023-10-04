install:
	pipenv install
	pipenv requirements > requirements.txt
	tempfile=$(mktemp)
	sed '1d' requirements.txt > "$tempfile"
	mv "$tempfile" requirements.txt
	pip install -e ".[dev]"

lint:
	pylint --disable=all --enable=unused-import pysurvival

fmt:
	isort -rc .
	black . --line-length 99

test:
	pytest .