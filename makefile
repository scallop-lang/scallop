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

test-cargo:
	cargo test --workspace

test-scallopy:
	python3 etc/scallopy/tests/test.py
