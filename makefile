all:
	@echo "Try `make install-scli`, `make install-sclc`, or `make install-scallopy`"

install-scli:
	cargo install --path etc/scli

install-sclc:
	cargo install --path etc/sclc

install-sclrepl:
	cargo install --path etc/sclrepl

install-scallop-cli: install-scallopy
	cd etc/scallop-cli; python setup.py install

develop-scallop-cli: develop-scallopy
	cd etc/scallop-cli; python setup.py install

install-scallopy:
	maturin build --release --manifest-path etc/scallopy/Cargo.toml --out target/wheels/current
	find target/wheels/current -name "*.whl" -print | xargs pip install --force-reinstall
	rm -rf target/wheels/current

develop-scallopy:
	cd etc/scallopy; maturin develop --release

# ============================================
# === Scallopy with Torch on normal Device ===

install-scallopy-torch:
	maturin build --release \
		--features "torch-tensor" \
		--manifest-path etc/scallopy/Cargo.toml \
		--out target/wheels/current \
		--config 'env.LIBTORCH_USE_PYTORCH = "1"'
	find target/wheels/current -name "*.whl" -print | xargs pip install --force-reinstall
	rm -rf target/wheels/current

develop-scallopy-torch:
	cd etc/scallopy; maturin develop --release --features "torch-tensor" --config 'env.LIBTORCH_USE_PYTORCH = "1"'

# =================================================
# === Scallopy with Torch on Apple M1/M2 Device ===

install-scallopy-torch-apple:
	python3 scripts/link_torch_lib.py
	maturin build --release \
		--features "torch-tensor" \
		--manifest-path etc/scallopy/Cargo.toml \
		--out target/wheels/current \
		--config 'env.LIBTORCH = "$(shell pwd)/.tmp/torch"' \
		--config 'env.DYLD_LIBRARY_PATH = "$(shell pwd)/.tmp/torch/lib"'
	find target/wheels/current -name "*.whl" -print | xargs pip install --force-reinstall
	rm -rf target/wheels/current

develop-scallopy-torch-apple:
	python3 scripts/link_torch_lib.py
	cd etc/scallopy; maturin develop --release \
		--features "torch-tensor" \
		--config 'env.LIBTORCH = "$(shell pwd)/.tmp/torch"' \
		--config 'env.DYLD_LIBRARY_PATH = "$(shell pwd)/.tmp/torch/lib"'

# =================================================
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

check-plus:
	cargo check --workspace --features "torch-tensor"

test:
	@echo "[Info] Performing cargo test..."
	@make test-cargo
	@echo "[Info] Performing scallopy test..."
	@make test-scallopy

test-all:
	@echo "[Info] Performing cargo test..."
	@make test-cargo
	@echo "[Info] Performing cargo test [ignored]..."
	@make test-cargo-ignored
	@echo "[Info] Performing scallopy test..."
	@make test-scallopy

test-cargo:
	cargo test --workspace

test-cargo-ignored:
	cargo test --workspace -- --ignored

test-scallopy: develop-scallopy
	python3 etc/scallopy/tests/test.py

test-scallopy-torch: develop-scallopy-torch
	python3 etc/scallopy/tests/test.py

doc:
	cargo doc

serve-doc:
	@echo "Starting Scallop Rust documentation server on port 8192..."
	@cd target/doc; python3 -m http.server 8192 > /dev/null 2>&1 & open http://localhost:8192/scallop_core

stop-serve-doc:
	@echo "Stopping documentation server on port 8192..."
	@lsof -t -i:8192 | xargs kill

serve-book:
	mdbook serve -p 8193 doc/

stop-serve-book:
	@echo "Stopping book server on port 8193..."
	@lsof -t -i:8193 | xargs kill
