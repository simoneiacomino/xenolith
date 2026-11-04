# Build Xenolith from source

Commands run from the repository root. On Arch Linux, install the build tools
and Level Zero runtime dependencies:

```sh
sudo pacman -S --needed gcc make binutils coreutils level-zero-headers level-zero-loader intel-compute-runtime
```

Get Unsloth's [Gemma 4 26B-A4B IT QAT UD-Q4_K_XL GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/main/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf), then build and configure Xenolith:

```sh
make
./xenolith config init --model /absolute/path/model.gguf --ctx 65536
./xenolith chat
```

Configuration is shared with packaged installations. Initialization does not
download a model or overwrite an existing configuration. Add
`--mmproj /absolute/path/mmproj-F16.gguf` to `config init` to enable vision.
See the [CLI reference](cli.md#native-configuration) for configuration paths
and options.

To install the built command for your user:

```sh
make install
```

By default, `make install` places `xenolith` under `~/.local/bin`. Set
`PREFIX=/chosen/root` to use another installation prefix. Packagers can stage
the command with `make install PREFIX=/usr DESTDIR=/absolute/staging`; `DESTDIR`
is prepended to the install path. The target installs only the `xenolith`
command and refuses to replace a symlink or directory at that command path.

For a Level Zero device/runtime check that does not load model weights, see
[`xenolith check-device`](cli.md#device-check). For all commands and options,
run `./xenolith --help`.
