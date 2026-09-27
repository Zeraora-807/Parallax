"""Bootstrap a project-local environment and launch the Parallax CLI."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "start.local.json"
IMAGE = "parallax-tools:latest"
CLI = "from parallax.cli import main; raise SystemExit(main())"
SKILL_ROOT = ROOT / "skills"
MAX_MODEL_LIST_BYTES = 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def ui(stream=None):
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from parallax.ui import AfterglowUI

    return AfterglowUI(stream or sys.stdout)


def prepare_environment() -> Path:
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        ui().status("env", "creating .venv", state="warning")
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    version = subprocess.run(
        [str(python), "-c", "import sys; sys.exit(sys.version_info < (3, 11))"],
        check=False,
    )
    if version.returncode:
        raise ValueError(".venv 无法使用 Python 3.11+；请重命名旧 .venv 后重试。")
    fingerprint = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
    marker = environment / ".parallax-bootstrap"
    probe = subprocess.run(
        [str(python), "-c", "import parallax.cli, httpx, jsonschema"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if (
        probe.returncode
        or not marker.is_file()
        or marker.read_text(encoding="utf-8") != fingerprint
    ):
        ui().status("env", "installing dependencies", state="warning")
        subprocess.run([str(python), "-m", "pip", "install", "-e", str(ROOT)], cwd=ROOT, check=True)
        marker.write_text(fingerprint, encoding="utf-8")
    return python


def ask(label: str, default: str = "", *, required: bool = True, secret: bool = False) -> str:
    while True:
        prompt = ui().prompt(label, default)
        value = (getpass.getpass(prompt) if secret else input(prompt)).strip() or default
        if value or not required:
            return value
        ui().status(label, "required", state="warning")


def read_config() -> dict[str, str]:
    if not CONFIG.is_file():
        return {}
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in config.items()
        ):
            raise ValueError("配置必须是字符串键值对象")
        return config
    except (ValueError, OSError) as error:
        raise ValueError(f"无法读取 {CONFIG.name}；请修正或重命名该文件：{error}") from error


def discover_models(base_url: str, api_key: str) -> tuple[list[str], str | None]:
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Accept": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="GET",
    )
    try:
        with build_opener(_NoRedirect).open(request, timeout=15) as response:
            content = response.read(MAX_MODEL_LIST_BYTES + 1)
        if len(content) > MAX_MODEL_LIST_BYTES:
            return [], "response too large"
        payload = json.loads(content)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return [], "invalid response"
        models = []
        for item in data:
            model = item.get("id") if isinstance(item, dict) else None
            if isinstance(model, str) and 0 < len(model.strip()) <= 512:
                models.append(model.strip())
        return list(dict.fromkeys(models)), None if models else "empty list"
    except HTTPError as error:
        return [], f"HTTP {error.code}"
    except (URLError, OSError, TimeoutError, ValueError) as error:
        return [], type(error).__name__


def select_skill_files(config: dict[str, str]) -> list[Path]:
    available = {
        path.stem: path
        for path in sorted(SKILL_ROOT.glob("*.md"))
        if path.is_file()
        and not path.is_symlink()
        and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", path.stem)
    }
    if not available:
        ui().status("skill", "none")
        return []
    choices = "|".join(["none", *available])
    selected = ask(f"Skill <{choices}>", config.get("skills", "none"), required=False).strip()
    if not selected or selected.casefold() == "none":
        return []
    names = list(dict.fromkeys(name.strip() for name in selected.split(",") if name.strip()))
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"未知 Skill：{unknown[0]}")
    return [available[name] for name in names]


def prepare_docker() -> None:
    docker = shutil.which("docker")
    if docker is None:
        raise ValueError("未找到 Docker。请安装并启动 Docker Desktop，使用 Linux containers。")
    info = subprocess.run(
        [docker, "info", "--format", "{{.OSType}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if info.returncode:
        raise ValueError("Docker 服务不可用。请启动 Docker Desktop，等待就绪后重试。")
    if info.stdout.strip() != "linux":
        raise ValueError("请将 Docker 切换为 Linux containers 后重试。")
    inspect = subprocess.run(
        [docker, "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    if inspect.returncode:
        ui().status("docker", "building tools image", state="warning")
        try:
            subprocess.run(
                [docker, "build", "-f", "docker/tools.Dockerfile", "-t", IMAGE, "."],
                cwd=ROOT,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"分析工具镜像构建失败（退出码 {error.returncode}），请查看上方 Docker 输出。\n"
                "若连接 auth.docker.io 或 registry-1.docker.io 超时，请检查网络以及 Docker Desktop\n"
                "的 Settings > Resources > Proxies 设置，确认 Docker 能使用可用的代理。\n"
                "先运行 docker pull debian:bookworm-slim；成功后重新启动 start.bat。\n"
                "若失败发生在 apt-get 阶段，请检查容器构建时到 Debian 软件源的连接。"
            ) from error


def interactive_arguments() -> list[str]:
    config = read_config()
    from parallax import __version__

    print(ui().banner(__version__))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        ui().status("API key", "environment", state="success")
    else:
        api_key = ask("API key", required=False, secret=True)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    base_url = ask(
        "Endpoint",
        os.environ.get("OPENAI_BASE_URL") or config.get("base_url", "http://127.0.0.1:8000/v1"),
    )
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("模型接口地址必须是有效的 http:// 或 https:// URL。")
    if endpoint.username is not None or endpoint.password is not None:
        raise ValueError("接口地址不能包含用户名或密码；请使用 API key。")
    models, discovery_error = discover_models(base_url, api_key)
    preferred_model = os.environ.get("OPENAI_MODEL") or config.get("model", "")
    if models:
        model = preferred_model if preferred_model in models else models[0]
        ui().status("model", model, state="success")
    else:
        ui().status("model", f"auto-detect failed: {discovery_error}", state="warning")
        model = ask("Model", preferred_model)
    sample = Path(ask("Sample", config.get("sample", "")).strip('"'))
    sample = sample.expanduser().resolve(strict=True)
    if not sample.is_file():
        raise ValueError("样本路径必须指向一个文件。")
    workspace = (
        Path(ask("Workspace", config.get("workspace", str(sample.parent))).strip('"'))
        .expanduser()
        .resolve(strict=True)
    )
    if not workspace.is_dir():
        raise ValueError("工作目录必须是一个已存在的目录。")
    try:
        relative = sample.relative_to(workspace)
    except ValueError as error:
        raise ValueError("样本必须位于工作目录内；请调整工作目录后重试。") from error
    if relative.parts[0].casefold() == ".parallax":
        raise ValueError("不能将 .parallax 状态目录中的文件作为输入样本。")
    prompt = ask("Task", config.get("prompt", "分析已授权样本的格式、保护与可疑字符串"))
    skill_files = select_skill_files(config)
    config = {
        "model": model,
        "base_url": base_url,
        "sample": str(sample),
        "workspace": str(workspace),
        "prompt": prompt,
        "skills": ",".join(path.stem for path in skill_files) or "none",
    }
    temporary = CONFIG.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(CONFIG)
    prepare_docker()
    arguments = [
        "run",
        "--workspace",
        str(workspace),
        "--input",
        relative.as_posix(),
        "--executor",
        "docker",
        "--docker-image",
        IMAGE,
        "--model",
        model,
        "--base-url",
        base_url,
    ]
    for skill_file in skill_files:
        arguments.extend(["--skill-file", str(skill_file)])
    return [*arguments, "--", prompt]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if sys.version_info < (3, 11):
        print("[错误] 请安装 Python 3.11 或更新版本。", file=sys.stderr)
        return 1
    os.environ["PYTHONUTF8"] = "1"
    try:
        python = prepare_environment()
        arguments = sys.argv[1:] or interactive_arguments()
        return subprocess.run(
            [str(python), "-c", CLI, *arguments], cwd=ROOT, check=False
        ).returncode
    except (KeyboardInterrupt, EOFError):
        ui().status("run", "cancelled", state="warning")
        return 130
    except subprocess.CalledProcessError as error:
        ui(sys.stderr).error(f"环境准备失败（退出码 {error.returncode}）")
        return error.returncode
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        ui(sys.stderr).error(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
