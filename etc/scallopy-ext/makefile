.PHONY: build install develop

build:
	python -m build

install: build
	find dist -name "*.whl" -print | xargs pip install --force-reinstall

develop:
	pip install --editable .

clean:
	rm -rf dist
