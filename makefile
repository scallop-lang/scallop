distr_dir = target/distribute

all:
	@echo "Try `make install-scli`, `make install-sclc`, or `make install-scallopy`"

# ================================
# === Base Scallop Executables ===

install-scli:
	cargo install --path etc/scli

build-scli:
	cargo build --release --bin scli

install-sclc:
	cargo install --path etc/sclc

build-sclc:
	cargo build --release --bin sclc

install-sclrepl:
	cargo install --path etc/sclrepl

build-sclrepl:
	cargo build --release --bin sclrepl

# ===============================
# === Python related packages ===

## Scallopy

install-scallopy:
	maturin build --release --manifest-path etc/scallopy/Cargo.toml --out target/wheels/current
	find target/wheels/current -name "*.whl" -print | xargs pip install --force-reinstall
	rm -rf target/wheels/current

build-scallopy:
	maturin build --release --manifest-path etc/scallopy/Cargo.toml --out target/wheels

develop-scallopy:
	cd etc/scallopy; maturin develop --release

## Scallopy Extension Lib

install-scallopy-ext: install-scallopy
	make -C etc/scallopy-ext install

build-scallopy-ext: develop-scallopy
	make -C etc/scallopy-ext build

develop-scallopy-ext: develop-scallopy
	make -C etc/scallopy-ext develop

## Scallop CLI

install-scallop-cli: install-scallopy-ext
	make -C etc/scallop-cli install

build-scallop-cli: build-scallopy-ext
	make -C etc/scallop-cli build

develop-scallop-cli: develop-scallopy-ext
	make -C etc/scallop-cli develop

## Scallopy Plugins

install-scallopy-plugins: install-scallopy-ext
	make -C etc/scallopy-plugins install

build-scallopy-plugins: build-scallopy-ext
	make -C etc/scallopy-plugins build

develop-scallopy-plugins: develop-scallopy-ext
	make -C etc/scallopy-plugins develop

# =================================================
# === Collect wheels and packages to distribute ===

distribute:
	@echo "==> Create distribute folder..."
	mkdir -p $(distr_dir)

	@echo "==> Copy core scallop..."
	cp target/release/scli $(distr_dir)
	cp target/release/sclrepl $(distr_dir)
	cp target/release/sclc $(distr_dir)

	@echo "==> Copy wheels..."
	cp target/wheels/*.whl $(distr_dir)
	cp etc/scallopy-ext/dist/*.whl $(distr_dir)
	cp etc/scallop-cli/dist/*.whl $(distr_dir)
	cp -r etc/scallopy-plugins/**/dist/*.whl $(distr_dir)

# ==========================================
# === Scallop WASM for Web Demo and Node ===

wasm-demo:
	make -C etc/scallop-wasm

run-wasm-demo:
	cd etc/scallop-wasm/demo; python3 -m http.server

# ==========================================
# === Scallop vitrual environment ===

init-venv:
	python3 -m venv .env
	.env/bin/pip install --upgrade pip
	.env/bin/pip install maturin torch torchvision transformers gym scikit-learn opencv-python tqdm matplotlib

clear-venv:
	rm -rf .env

# =============================
# === Scallop VSCode Plugin ===

vscode-plugin:
	make -C etc/vscode-scl

# ====================================
# === Project testing and cleaning ===

clean:
	cargo clean
	make -C etc/scallopy clean
	make -C etc/scallopy-ext clean
	make -C etc/scallopy-plugins clean
	make -C etc/scallop-cli clean
	make -C etc/scallop-wasm clean

check:
	cargo check --workspace

test:
	@echo "[Info] Performing cargo test..."
	@make test-cargo
	@echo "[Info] Performing scallopy test..."
	@make test-scallopy
	@echo "[Info] Performing scallopy-ext test..."
	@make test-scallopy-ext

test-all: test
	@echo "[Info] Performing cargo test [ignored]..."
	@make test-cargo-ignored
	@echo "[Info] Performing scallopy test..."
	@make test-scallopy
	@echo "[Info] Performing scallopy-ext test..."
	@make test-scallopy-ext

test-cargo:
	cargo test --workspace

test-cargo-ignored:
	cargo test --workspace -- --ignored

test-scallopy: develop-scallopy
	python3 etc/scallopy/tests/test.py

test-scallopy-ext: develop-scallopy-ext
	python3 etc/scallopy-ext/tests/test.py

# ======================
# === Documentations ===

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
