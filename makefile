all:
	@echo "Try `make install-scli`, `make install-sclc`, or `make install-scallopy`"

install-scli:
	cargo install --path etc/scli

install-sclc:
	cargo install --path etc/sclc

install-sclrepl:
	cargo install --path etc/sclrepl

install-scallopy:
	maturin build --release --manifest-path etc/scallopy/Cargo.toml --out target/wheels/current
	find target/wheels/current -name "*.whl" -print | xargs pip install --force-reinstall
	rm -rf target/wheels/current

develop-scallopy:
	cd etc/scallopy; maturin develop --release

wasm-demo:
	make -C etc/scallop-wasm

run-wasm-demo:
	cd etc/scallop-wasm/demo; python3 -m http.server

py-venv:
	python3 -m venv .env

vscode-plugin:
	make -C etc/vscode-scl

clean:
	cargo clean
	make -C etc/scallopy clean
	make -C etc/scallop-wasm clean

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

test-scallopy:
	python3 etc/scallopy/tests/test.py

doc:
	cargo doc

serve-doc:
	@echo "Starting Scallop Rust documentation server on port 8192..."
	@cd target/doc; python3 -m http.server 8192 > /dev/null 2>&1 & open http://localhost:8192/scallop_core

stop-serve-doc:
	@echo "Stopping documentation server on port 8192..."
	@lsof -t -i:8192 | xargs kill
