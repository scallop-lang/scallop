all:
	@echo "Try `make install-scli`, `make install-sclc`, or `make install-scallopy`"

install-scli:
	cargo install --path etc/scli

install-sclc:
	cargo install --path etc/sclc

install-sclrepl:
	cargo install --path etc/sclrepl

install-scallop-cli: install-scallopy-ext
	make internal-install-scallop-cli

develop-scallop-cli: develop-scallopy-ext
	make internal-develop-scallop-cli

install-scallopy-ext: install-scallopy
	make internal-install-scallopy-ext

develop-scallopy-ext: develop-scallopy
	make internal-develop-scallopy-ext

install-scallopy:
	maturin build --release --manifest-path etc/scallopy/Cargo.toml --out target/wheels/current
	find target/wheels/current -name "*.whl" -print | xargs pip install --force-reinstall
	rm -rf target/wheels/current

develop-scallopy:
	cd etc/scallopy; maturin develop --release

# ====================================================================
# === Internal installing scripts for scallop-cli and scallopy-ext ===

internal-install-scallop-cli:
	cd etc/scallop-cli; python -m build
	find etc/scallop-cli/dist -name "*.whl" -print | xargs pip install --force-reinstall

internal-develop-scallop-cli:
	cd etc/scallop-cli; pip install --editable .

internal-install-scallopy-ext:
	cd etc/scallopy-ext; python -m build
	find etc/scallopy-ext/dist -name "*.whl" -print | xargs pip install --force-reinstall

internal-develop-scallopy-ext:
	cd etc/scallopy-ext; pip install --editable .

# ==========================================
# === Scallop WASM for Web Demo and Node ===

wasm-demo:
	make -C etc/scallop-wasm

run-wasm-demo:
	cd etc/scallop-wasm/demo; python3 -m http.server

init-venv:
	python3 -m venv .env
	.env/bin/pip install --upgrade pip
	.env/bin/pip install maturin torch torchvision transformers gym scikit-learn opencv-python tqdm matplotlib

clear-venv:
	rm -rf .env

vscode-plugin:
	make -C etc/vscode-scl

clean:
	cargo clean
	make -C etc/scallopy clean
	make -C etc/scallop-wasm clean

check:
	cargo check --workspace

check-torch:
	cargo check --workspace --features "torch-tensor"

test:
	@echo "[Info] Performing cargo test..."
	@make test-cargo
	@echo "[Info] Performing scallopy test..."
	@make test-scallopy
	@echo "[Info] Performing scallopy-ext test..."
	@make test-scallopy-ext

test-torch:
	@echo "[Info] Performing cargo test..."
	@make test-cargo
	@echo "[Info] Performing scallopy test..."
	@make test-scallopy-torch
	@echo "[Info] Performing scallopy-ext test..."
	@make test-scallopy-ext-torch

test-all: test
	@echo "[Info] Performing cargo test [ignored]..."
	@make test-cargo-ignored

test-all-torch: test-torch
	@echo "[Info] Performing cargo test [ignored]..."
	@make test-cargo-ignored

test-cargo:
	cargo test --workspace

test-cargo-ignored:
	cargo test --workspace -- --ignored

test-scallopy: develop-scallopy
	python3 etc/scallopy/tests/test.py

test-scallopy-torch: develop-scallopy-torch
	python3 etc/scallopy/tests/test.py

test-scallopy-ext: develop-scallopy-ext
	python3 etc/scallopy-ext/tests/test.py

test-scallopy-ext-torch: develop-scallopy-ext-torch
	python3 etc/scallopy-ext/tests/test.py

doc:
	cargo doc

serve-doc:
	@echo "Starting Scallop Rust documentation server on port 8192..."
	@cd target/doc; python3 -m http.server 8192 > /dev/null 2>&1 & open http://localhost:8192/scallop_core

stop-serve-doc:
	@echo "Stopping documentation server on port 8192..."
	@lsof -t -i:8192 | xargs kill

build-book:
	mdbook build doc/

serve-book:
	mdbook serve -p 8193 doc/

stop-serve-book:
	@echo "Stopping book server on port 8193..."
	@lsof -t -i:8193 | xargs kill
