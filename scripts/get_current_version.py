import toml
file = open("core/Cargo.toml", "r")
parsed_toml = toml.loads(file.read())
print(parsed_toml["package"]["version"])
