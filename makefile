all:
	@echo "Try `make install-scli`, `make install-sclc`, or `make install-scallopy`

install-scli:
	cargo install --path etc/scli

install-sclc:
	cargo install --path etc/sclc

install-sclrepl:
	cargo install --path etc/sclrepl

install-scallopy:
	cd etc/scallopy; maturin develop --release

setup-py-venv:
	python3 -m venv .env

setup-wasm-demo:
	make -C etc/scallop-wasm

run-wasm-demo:
	cd etc/scallop-wasm/demo; python3 -m http.server

clean:
	cargo clean
