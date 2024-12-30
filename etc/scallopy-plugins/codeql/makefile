develop:
	pip install --editable .

install: build
	find dist -name "*.whl" -print | xargs pip install --force-reinstall

build:
	python -m build

test:
	python tests/test.py

clean:
	rm -rf dist
