"""Dependency-light local HTTP server for the Parallax WebUI."""

from __future__ import annotations

import json
import re
import sys
import threading
import uuid
import webbrowser
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

from parallax.executor.web import run_agent_process
from parallax.harness.user_agents import MAX_USER_AGENT_BYTES, load_user_agent
from parallax.memory import MemoryStore
from parallax.web.plugins import (
    ContextPlugin,
    ConversationHistoryPlugin,
    MemoryPlugin,
    ReferenceConversationPlugin,
)
from parallax.web.store import WebStore


STATIC_ROOT = Path(__file__).with_name("static")
MAX_JSON = 1024 * 1024
MAX_UPLOAD = 64 * 1024 * 1024


class RunCoordinator:
    def __init__(
        self,
        data_root: Path,
        store: WebStore,
        memory: MemoryStore,
        skill_root: Path,
        agent_root: Path,
    ) -> None:
        self.data_root = data_root
        self.store = store
        self.skill_root = skill_root
        self.agent_root = agent_root
        self.plugins: list[ContextPlugin] = [
            MemoryPlugin(memory),
            ReferenceConversationPlugin(store),
            ConversationHistoryPlugin(store),
        ]
        self._lock = threading.Lock()
        self._running: set[str] = set()

    def start(self, conversation_id: str, payload: dict[str, object]) -> str:
        content = str(payload.get("content", "")).strip()
        if not content or len(content) > 20_000:
            raise ValueError("message must contain 1–20000 characters")
        if not self.store.conversation(conversation_id):
            raise ValueError("对话不存在")
        if not str(payload.get("model", "")).strip() or not _valid_url(
            str(payload.get("base_url", ""))
        ):
            raise ValueError("请先在设置中填写接口地址和模型")
        with self._lock:
            if conversation_id in self._running:
                raise ValueError("conversation already has a running response")
            if len(self._running) >= 2:
                raise ValueError("已有两项分析在运行，请稍后再试")
            self._running.add(conversation_id)
        response_id = None
        try:
            response_id = self.store.begin_turn(conversation_id, content)
            thread = threading.Thread(
                target=self._run, args=(conversation_id, response_id, content, payload), daemon=True
            )
            thread.start()
        except Exception:
            with self._lock:
                self._running.discard(conversation_id)
            if response_id:
                self.store.update_message(response_id, "无法启动分析，请重试", "error")
            raise
        return response_id

    def _run(
        self, conversation_id: str, response_id: str, content: str, payload: dict[str, object]
    ) -> None:
        temporary_files: list[Path] = []
        try:
            workspace = self.data_root / "workspaces" / conversation_id
            workspace.mkdir(parents=True, exist_ok=True)
            sections = [plugin.context(conversation_id, content) for plugin in self.plugins]
            prompt = "\n\n".join(
                [
                    *filter(None, sections),
                    "<current-user-message>\n" + content + "\n</current-user-message>",
                ]
            )
            model = str(payload.get("model", "")).strip()
            base_url = str(payload.get("base_url", "")).strip()
            if not model or not _valid_url(base_url):
                raise ValueError("valid model and base_url are required")
            temporary_root = self.data_root / "tmp"
            temporary_root.mkdir(parents=True, exist_ok=True)
            temporary_id = uuid.uuid4().hex
            prompt_path = temporary_root / f"{temporary_id}.prompt.txt"
            memory_source_path = temporary_root / f"{temporary_id}.source.txt"
            temporary_files.extend([prompt_path, memory_source_path])
            prompt_path.write_text(prompt, encoding="utf-8")
            memory_source_path.write_text(content, encoding="utf-8")
            command = [
                sys.executable,
                "-c",
                "from parallax.cli import main; raise SystemExit(main())",
                "run",
                "--workspace",
                str(workspace),
                "--executor",
                "docker",
                "--model",
                model,
                "--base-url",
                base_url,
            ]
            if payload.get("multi_agent") is True:
                raw_max_agents = payload.get("max_agents", 4)
                if isinstance(raw_max_agents, bool) or not isinstance(raw_max_agents, int):
                    raise ValueError("max_agents must be an integer between 2 and 8")
                max_agents = raw_max_agents
                if not 2 <= max_agents <= 8:
                    raise ValueError("max_agents must be an integer between 2 and 8")
                command.extend(["--multi-agent", "--max-agents", str(max_agents)])
            for attachment in self.store.attachments(conversation_id):
                command.extend(["--input", Path(attachment["path"]).name])
            selected = payload.get("skills", [])
            if isinstance(selected, list):
                available = {item["name"]: item["path"] for item in self.list_skills()}
                for name in selected:
                    if isinstance(name, str) and name in available:
                        command.extend(["--skill-file", available[name]])
            selected_agents = payload.get("agents", [])
            if isinstance(selected_agents, list):
                available_agents = {item["name"]: item["path"] for item in self.list_agents()}
                enabled_agents = [
                    available_agents[name]
                    for name in selected_agents
                    if isinstance(name, str) and name in available_agents
                ]
                if enabled_agents and payload.get("multi_agent") is not True:
                    raise ValueError("custom Agents require multi-agent mode")
                for path in enabled_agents:
                    command.extend(["--agent-file", path])
            for plugin in self.plugins:
                command.extend(plugin.run_arguments(conversation_id, content))
            command.extend(
                [
                    "--prompt-file",
                    str(prompt_path),
                    "--memory-source-file",
                    str(memory_source_path),
                    "--",
                    "web-context",
                ]
            )
            key = str(payload.get("api_key", ""))
            # An empty browser key must not inherit a credential for another endpoint.
            returncode, stdout, stderr = run_agent_process(
                command, workspace, {"OPENAI_API_KEY": key}
            )
            if returncode:
                detail = [line.strip() for line in stderr.splitlines() if line.strip()]
                message = next(
                    (line for line in detail if line.removeprefix("[red]").startswith("ERROR")),
                    detail[-1] if detail else "analysis failed",
                )
                raise ValueError(message)
            self.store.update_message(response_id, stdout.strip(), "done")
        except Exception as error:
            self.store.update_message(response_id, str(error), "error")
        finally:
            for path in temporary_files:
                with suppress(OSError):
                    path.unlink(missing_ok=True)
            with self._lock:
                self._running.discard(conversation_id)

    def list_skills(self) -> list[dict[str, object]]:
        result = []
        self.skill_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.skill_root.glob("*.md")):
            size = path.stat().st_size if path.is_file() and not path.is_symlink() else None
            if (
                size is not None
                and size <= 64 * 1024
                and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", path.stem)
            ):
                result.append(
                    {
                        "name": path.stem,
                        "path": str(path),
                        "bytes": size,
                    }
                )
        return result

    def list_agents(self) -> list[dict[str, object]]:
        result = []
        self.agent_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.agent_root.glob("*.agent.toml")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size > MAX_USER_AGENT_BYTES
            ):
                continue
            try:
                agent = load_user_agent(path)
            except Exception:
                continue
            result.append(
                {
                    "name": path.name.removesuffix(".agent.toml"),
                    "path": str(path),
                    "role": agent.contribution.profiles[0].role,
                    "digest": agent.digest,
                }
            )
        return result


class WebApplication:
    def __init__(self, data_root: Path, skill_root: Path, agent_root: Path | None = None) -> None:
        data_root.mkdir(parents=True, exist_ok=True)
        agent_root = agent_root or skill_root.parent / "agents"
        self.store = WebStore(data_root / "webui.sqlite3")
        self.memory = MemoryStore(self.store.path)
        self.coordinator = RunCoordinator(
            data_root, self.store, self.memory, skill_root, agent_root
        )
        self.skill_root = skill_root
        self.agent_root = agent_root


def make_handler(application: WebApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ParallaxWeb/0.3"

        def log_message(self, format, *args):
            return

        def do_GET(self):
            try:
                self._same_origin()
                self._get()
            except Exception as error:
                self._error(error)

        def do_POST(self):
            try:
                self._same_origin()
                self._post()
            except Exception as error:
                self._error(error)

        def do_DELETE(self):
            try:
                self._same_origin()
                self._delete()
            except Exception as error:
                self._error(error)

        def _get(self):
            path = urlsplit(self.path).path
            if path == "/api/state":
                return self._json(
                    {
                        "conversations": application.store.list_conversations(),
                        "settings": application.store.settings(),
                        "skills": application.coordinator.list_skills(),
                        "agents": application.coordinator.list_agents(),
                        "memories": application.memory.list(),
                    }
                )
            match = re.fullmatch(r"/api/conversations/([0-9a-f]{32})", path)
            if match:
                conversation = application.store.conversation(match.group(1))
                if not conversation:
                    return self._json({"error": "not found"}, 404)
                conversation["messages"] = application.store.messages(match.group(1))
                conversation["attachments"] = application.store.attachments(match.group(1))
                return self._json(conversation)
            target = "index.html" if path in {"/", "/index.html"} else path.removeprefix("/")
            if target not in {
                "index.html",
                "app.js",
                "styles.css",
                "api.js",
                "ui.js",
                "settings.js",
            }:
                return self.send_error(404)
            content = (STATIC_ROOT / target).read_bytes()
            mime = {
                "html": "text/html; charset=utf-8",
                "js": "text/javascript; charset=utf-8",
                "css": "text/css; charset=utf-8",
            }[target.rsplit(".", 1)[1]]
            self.send_response(200)
            self._headers(mime)
            self.end_headers()
            self.wfile.write(content)

        def _post(self):
            path = urlsplit(self.path).path
            if path == "/api/conversations":
                body = self._body_json()
                conversation_id = application.store.create_conversation(
                    str(body.get("title", "")), _string_list(body.get("reference_ids", []))
                )
                return self._json({"id": conversation_id}, 201)
            match = re.fullmatch(r"/api/conversations/([0-9a-f]{32})/messages", path)
            if match:
                response_id = application.coordinator.start(match.group(1), self._body_json())
                return self._json({"response_id": response_id}, 202)
            match = re.fullmatch(r"/api/conversations/([0-9a-f]{32})/attachments", path)
            if match:
                return self._upload(match.group(1))
            if path == "/api/settings":
                body = self._body_json()
                application.store.save_settings({str(k): str(v) for k, v in body.items()})
                return self._json({"saved": True})
            if path == "/api/skills":
                body = self._body_json()
                name = str(body.get("name", ""))
                content = str(body.get("content", ""))
                if (
                    not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                    or not 0 < len(content.encode()) <= 64 * 1024
                ):
                    raise ValueError("invalid skill name or content")
                application.skill_root.mkdir(parents=True, exist_ok=True)
                if name.upper() in {
                    "CON",
                    "PRN",
                    "AUX",
                    "NUL",
                    *[f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)],
                }:
                    raise ValueError("此 Skill 名称是系统保留名称")
                try:
                    with (application.skill_root / f"{name}.md").open(
                        "x", encoding="utf-8"
                    ) as output:
                        output.write(content)
                except FileExistsError:
                    raise ValueError("同名 Skill 已存在，请使用其他名称") from None
                return self._json({"saved": True}, 201)
            if path == "/api/agents":
                body = self._body_json()
                name = str(body.get("name", ""))
                content = str(body.get("content", ""))
                if (
                    not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                    or not 0 < len(content.encode()) <= MAX_USER_AGENT_BYTES
                ):
                    raise ValueError("invalid Agent name or content")
                application.agent_root.mkdir(parents=True, exist_ok=True)
                target = application.agent_root / f"{name}.agent.toml"
                try:
                    with target.open("x", encoding="utf-8") as output:
                        output.write(content)
                    load_user_agent(target)
                except FileExistsError:
                    raise ValueError("同名 Agent 已存在，请使用其他名称") from None
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                return self._json({"saved": True}, 201)
            if path == "/api/models":
                body = self._body_json()
                base = str(body.get("base_url", ""))
                key = str(body.get("api_key", ""))
                if not _valid_url(base):
                    raise ValueError("invalid base_url")
                with httpx.Client(follow_redirects=False, timeout=15) as client:
                    with client.stream(
                        "GET",
                        f"{base.rstrip('/')}/models",
                        headers={"Authorization": f"Bearer {key}"} if key else {},
                    ) as response:
                        content = bytearray()
                        for chunk in response.iter_bytes():
                            content.extend(chunk)
                            if len(content) > MAX_JSON:
                                raise ValueError("model list response too large")
                        response.raise_for_status()
                decoded = json.loads(content)
                data = decoded.get("data", []) if isinstance(decoded, dict) else []
                if not isinstance(data, list):
                    raise ValueError("模型列表格式无效，请手动输入模型名称")
                return self._json(
                    {
                        "models": [
                            item["id"]
                            for item in data
                            if isinstance(item, dict)
                            and isinstance(item.get("id"), str)
                            and 0 < len(item["id"].strip()) <= 512
                        ][:500]
                    }
                )
            return self.send_error(404)

        def _delete(self):
            match = re.fullmatch(r"/api/memories/([0-9a-f]{32})", urlsplit(self.path).path)
            if match:
                return self._json({"deleted": application.memory.delete(match.group(1))})
            return self.send_error(404)

        def _upload(self, conversation_id: str):
            if not application.store.conversation(conversation_id):
                raise ValueError("对话不存在")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD:
                raise ValueError("attachment must be 1–64 MiB")
            raw_name = parse_qs(urlsplit(self.path).query).get("name", ["sample.bin"])[0]
            name = (
                re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw_name)[-120:].strip(" .") or "sample.bin"
            )
            workspace = application.coordinator.data_root / "workspaces" / conversation_id
            workspace.mkdir(parents=True, exist_ok=True)
            target = workspace / uuid.uuid4().hex
            content = self.rfile.read(length)
            if len(content) != length:
                raise ValueError("incomplete attachment")
            try:
                target.write_bytes(content)
                attachment_id = application.store.add_attachment(conversation_id, name, target)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return self._json({"id": attachment_id, "name": name}, 201)

        def _body_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= MAX_JSON:
                raise ValueError("request body too large")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete JSON body")
            value = json.loads(raw or b"{}")
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _same_origin(self):
            port = self.server.server_address[1]
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host", "").lower()
            if host not in allowed_hosts:
                self._reject_origin("invalid Host header")
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                self._reject_origin("cross-origin request rejected")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._reject_origin("cross-site request rejected")

        def _reject_origin(self, reason):
            # Drain small rejected bodies so Windows does not reset the response socket.
            length = int(self.headers.get("Content-Length", "0"))
            if 0 < length <= MAX_JSON:
                self.connection.settimeout(2)
                with suppress(OSError):
                    self.rfile.read(length)
            raise ValueError(reason)

        def _json(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(content)))
            self._headers("application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)

        def _headers(self, content_type):
            self.send_header("Content-Type", content_type)
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; img-src 'self' data:; script-src 'self'; frame-ancestors 'none'",
            )

        def _error(self, error):
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def _valid_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and not parsed.query
            and not parsed.fragment
            and (parsed.port is None or 0 < parsed.port < 65536)
        )
    except ValueError:
        return False


def _string_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)][:20] if isinstance(value, list) else []


def serve(
    data_root: Path,
    skill_root: Path,
    host: str,
    port: int,
    open_browser: bool = True,
    *,
    agent_root: Path | None = None,
) -> None:
    application = WebApplication(
        data_root.resolve(),
        skill_root.resolve(),
        agent_root.resolve() if agent_root is not None else None,
    )
    server = ThreadingHTTPServer((host, port), make_handler(application))
    application.store.recover_interrupted()
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Parallax WebUI: {url}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
