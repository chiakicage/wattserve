# WattServe

## Set the environment

```shell
uv sync
source .venv/bin/activate # or other activate scripts for your own shell
python main.py --model /share/models/Qwen3-4B
```

If you want to change the index URL of pip in uv, you can modify your `~/.config/uv/uv.toml` as follows:

```toml
index-url = "https://mirrors.zju.edu.cn/pypi/web/simple"
```

## Develop Environment

```shell
uv sync --dev
pre-commit install
```
