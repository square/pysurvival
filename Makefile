install:
	pip install -e ".[dev]"

lint:
	pylint --disable=all --enable=unused-import pysurvival

fmt:
	isort -rc .
	black . --line-length 99

test:
	pytest .