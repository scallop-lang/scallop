DIRECTORY = $(patsubst %/, %, $(sort $(dir $(wildcard */.))))
DEVELOP_TARGETS = $(patsubst %,develop-%,$(DIRECTORY))
INSTALL_TARGETS = $(patsubst %,install-%,$(DIRECTORY))
BUILD_TARGETS = $(patsubst %,build-%,$(DIRECTORY))

develop: $(DEVELOP_TARGETS)

develop-%:
	cd $*; pip install --editable .

install: $(INSTALL_TARGETS)

install-%: build-%
	cd $*; find dist -name "*.whl" -print | xargs pip install --force-reinstall

build: $(BUILD_TARGETS)

build-%:
	cd $*; python -m build

clean:
	cd $*; rm -rf dist
