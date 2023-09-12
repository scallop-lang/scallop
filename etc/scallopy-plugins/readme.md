# Scallopy Plugins

This directory contains plugins written for `scallopy` for it to have capabilities interacting with other domain and models.
These include interfacing with GPT, vision models like SAM, multi-modal models like ViLT and CLIP, and so on.

## Install plugins

For the plugins listed in this current directory, one can easily install them using `make`.
Assuming you are under the root directory of this repository, you can do

``` bash
$ make -C etc/scallopy-plugins develop # to develop all plugins
# or
$ make -C etc/scallopy-plugins install # to install all plugins
# or
$ make -C etc/scallopy-plugins develop-<PLUGIN> # to develop the specific <PLUGIN>
# or
$ make -C etc/scallopy-plugins install-<PLUGIN> # to install the specific <PLUGIN>
```

Here, the difference between `develop` and `install` is that `develop` allows you do change the source code and run it without having to re-install everything.
Therefore, using `develop` is best suited if you are actively developing the plugins.
Otherwise, use `install` so that a concrete wheel (`.whl`) is built and installed into your environment.

For example, the `scallop-clip` plugin can be installed like the following:

``` bash
$ make -C etc/scallopy-plugins develop-clip
```

Note that for each installed plugin, the `scallop-ext` extension module will load them automatically, unless you specify that you want to include or exclude certain plugins (based on their names).

## Create new plugins

Please follow these steps to create a new plugin.

1. Create a new folder. If you want to create a new plugin in the source, create a folder under this directory.

2. In the folder you just created, add another file named `pyproject.toml`, and use the following template:

``` toml
[project]
name = "scallop-<YOUR-PLUGIN>"
version = "0.0.1"
dependencies = ["<YOUR-DEPENDENCY-1>", "<YOUR-DEPENDENCY-2>"]

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."scallop.plugin.setup_arg_parser"]
<NAME> = "scallop_<YOUR-PLUGIN>:setup_arg_parser"

[project.entry-points."scallop.plugin.configure"]
<NAME> = "scallop_<YOUR-PLUGIN>:configure"

[project.entry-points."scallop.plugin.load_into_context"]
<NAME> = "scallop_<YOUR-PLUGIN>:load_into_context"
```

Make sure you replace everything marked in angle brackets `<...>` with proper information.
For example, for the plugin for CLIP, we have the plugin named `scallop-clip`, and the `<NAME>` set to `clip`.
Here, the `<NAME>` will be used for Scallop plugin manager and users to decide wether to include or not include the plugin.
Therefore, try your best to name it well.

3. Under the folder you just created, create a new folder `src/scallop_<YOUR-PLUGIN>`.
Notice that there is an underscore (`"_"`) between `scallop` and your plugin name, instead of a `"-"`.

4. Under this folder, create a file named `__init__.py`. You can then add however many files under the same folder.

5. Make sure that your `__init__.py` exports three functions, they are

- `setup_arg_parser(parser: ArgumentParser)` for setting up command-line argument parsers
- `configure(args)` for configuring the plugin based on provided arguments
- `load_into_context(ctx: scallopy.Context)` that modifies the Scallop context.

If your plugin doesn't need any of the above three components, you can remove the corresponding item (`[project.entry-points."scallop.plugin.<COMPONENT>"]`) from the `pyproject.toml` file.
In this case, you don't need to export that function from your `__init__.py`.

Here, we provide some example of a simple plugin that loads a `key` from command-line arguments so that it can be accessed via a foreign function `$get_key()` inside of Scallop:

``` py
MY_KEY = None # a global variable holding the key

def setup_arg_parser(parser: ArgumentParser):
  parser.add_argument("--key", type=str, default="12345678")

def configure(args):
  global MY_KEY
  MY_KEY = args.key # load the key from cmd args

def load_into_context(ctx: scallopy.Context):
  # Create a foreign function that returns the loaded key
  @scallopy.foreign_function
  def get_key() -> str: return MY_KEY

  # Register the `$get_key` foreign function into ctx
  ctx.register_foreign_function(get_key)
```
