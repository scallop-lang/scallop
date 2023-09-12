DIRECTORY = $(patsubst %/, %, $(sort $(dir $(wildcard */.))))
DEVELOP_TARGETS = $(patsubst %,develop-%,$(DIRECTORY))
INSTALL_TARGETS = $(patsubst %,install-%,$(DIRECTORY))

develop: $(DEVELOP_TARGETS)

develop-%:
	cd $*; pip install --editable .

install: $(INSTALL_TARGETS)

install-%:
	cd $*; python -m build
	cd $*; find dist -name "*.whl" -print | xargs pip install --force-reinstall
