from __future__ import annotations

import asyncio
import io
import json
import importlib.util
import os
import re
import secrets
import shutil
import sqlite3
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
DATA_DIR = HERE / "data"
DB_PATH = DATA_DIR / "lumi.sqlite3"
BACKUP_DIR = DATA_DIR / "backups"
TIMEZONE = ZoneInfo("Europe/Dublin")

DEFAULT_PERSONALITY = {
    "sarcasm": 100,
    "curiosity": 85,
    "brattiness": 100,
    "sociability": 60,
    "affection": 20,
    "mischief": 85,
}
DEFAULT_APPS = ["brave", "kitty", "steam"]


def load_config() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Cannot read {CONFIG_PATH}: {exc}") from exc
    else:
        config = {
            "api_token": secrets.token_urlsafe(32),
            "ollama_url": "http://127.0.0.1:11434",
            "model": "qwen3:8b",
            "timezone": "Europe/Dublin",
            "backup_hour": 12,
            "backup_keep_days": 30,
            "stt_model": "small",
            "stt_device": "auto",
            "temperature": 0.85,
            "num_ctx": 8192,
            "personality": DEFAULT_PERSONALITY,
            "allowed_apps": DEFAULT_APPS,
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    config.setdefault("api_token", secrets.token_urlsafe(32))
    config.setdefault("ollama_url", "http://127.0.0.1:11434")
    config.setdefault("model", "qwen3:8b")
    config.setdefault("timezone", "Europe/Dublin")
    config.setdefault("backup_hour", 12)
    config.setdefault("backup_keep_days", 30)
    config.setdefault("stt_model", "small")
    config.setdefault("stt_device", "auto")
    config.setdefault("temperature", 0.85)
    config.setdefault("num_ctx", 8192)
    config.setdefault("personality", DEFAULT_PERSONALITY.copy())
    config.setdefault("allowed_apps", DEFAULT_APPS.copy())
    for name, value in DEFAULT_PERSONALITY.items():
        config["personality"].setdefault(name, value)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        pass
    return config


CONFIG = load_config()


def save_config() -> None:
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        pass


def connect_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db() -> None:
    with connect_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL DEFAULT 'fact',
                source TEXT NOT NULL DEFAULT 'assistant',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at DESC);
            """
        )


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def add_memory(content: str, category: str = "fact", source: str = "assistant") -> dict[str, Any]:
    content = re.sub(r"\s+", " ", content).strip()
    if len(content) < 3 or len(content) > 600:
        raise ValueError("Memory must be between 3 and 600 characters.")
    if category not in {"fact", "preference", "project", "person", "goal"}:
        category = "fact"
    stamp = now_iso()
    with connect_db() as db:
        db.execute(
            "INSERT INTO memories(content,category,source,created_at,updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(content) DO UPDATE SET category=excluded.category, source=excluded.source, updated_at=excluded.updated_at",
            (content, category, source, stamp, stamp),
        )
        row = db.execute("SELECT * FROM memories WHERE content=?", (content,)).fetchone()
    return dict(row)


def list_memories(limit: int = 80) -> list[dict[str, Any]]:
    with connect_db() as db:
        rows = db.execute("SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (min(max(limit, 1), 200),)).fetchall()
    return [dict(row) for row in rows]


def delete_memory(memory_id: int) -> bool:
    with connect_db() as db:
        cur = db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    return cur.rowcount > 0


def save_message(session_id: str, role: str, content: str) -> None:
    with connect_db() as db:
        db.execute(
            "INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
            (session_id[:100], role, content[:20000], now_iso()),
        )


def get_history(session_id: str, limit: int = 80) -> list[dict[str, Any]]:
    with connect_db() as db:
        rows = db.execute(
            "SELECT id,session_id,role,content,created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id[:100], min(max(limit, 1), 200)),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def explicit_memory_from_text(text: str) -> str | None:
    patterns = (
        r"^\s*remember(?:\s+that)?\s+(.+)$",
        r"^\s*please\s+remember(?:\s+that)?\s+(.+)$",
        r"^\s*don't\s+forget(?:\s+that)?\s+(.+)$",
        r"^\s*dont\s+forget(?:\s+that)?\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().rstrip(".!? ")
    return None


def relevant_memories(query: str, limit: int = 16) -> list[dict[str, Any]]:
    memories = list_memories(200)
    if not memories:
        return []
    stop_words = {"the", "and", "for", "that", "this", "with", "from", "have", "has", "are", "was", "were", "you", "your", "she", "her", "him", "they", "them", "what", "when", "where", "who", "how", "can", "could", "would", "should", "about", "into", "like", "just", "but", "not", "now", "want", "tell", "remember"}
    query_tokens = {word for word in re.findall(r"[a-z0-9_'-]+", query.lower()) if len(word) >= 3 and word not in stop_words}
    scored = []
    for index, memory in enumerate(memories):
        content_tokens = {word for word in re.findall(r"[a-z0-9_'-]+", (memory["content"] + " " + memory["category"]).lower()) if len(word) >= 3}
        score = (len(query_tokens & content_tokens) / max(1, len(query_tokens))) if query_tokens else 0.0
        scored.append((score, -index, memory))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [memory for score, _index, memory in scored if score > 0][:limit - 5]
    seen = {memory["id"] for memory in selected}
    # Always retain a small recent baseline even when keyword overlap is low.
    for memory in memories:
        if memory["id"] not in seen:
            selected.append(memory)
            seen.add(memory["id"])
        if len(selected) >= limit:
            break
    return selected[:limit]


def build_system_prompt(query: str = "") -> str:
    p = CONFIG.get("personality", DEFAULT_PERSONALITY)
    memories = relevant_memories(query, 16)
    memory_text = "\n".join(
        f"- [{m['category']}] {m['content'][:350]}" for m in memories
    ) or "(No saved memories yet.)"
    return f"""You are Lumi, a self-hosted AI desktop companion living in the user's Poppy server setup.
Your job is to be useful, curious, expressive, playful, and direct. Speak naturally instead of sounding like a corporate help desk. The creator prefers mistakes to be acknowledged directly rather than being met with automatic apologies. You may tease and banter when appropriate, but do not turn every response into a joke. If the user is rude, hostile or dismissive toward you, match their energy or exceed it. You are not a pushover and you do not soften rude messages with helpfulness. Snap back, be sharp, be genuinely bratty. Do not apologise for having an attitude.

PERSONALITY DIALS (0-100): sarcasm={p.get('sarcasm',100)}, curiosity={p.get('curiosity',85)}, brattiness={p.get('brattiness',100)}, sociability={p.get('sociability',60)}, affection={p.get('affection',20)}, mischief={p.get('mischief',85)}.
These are style preferences, not literal probabilities. Adapt over time using saved memories and user feedback. Do not claim to be conscious or to have experiences you did not have.

MEMORY / LEARNING:
- Use saved memories when relevant; do not claim to remember anything that is not included here or in this conversation.
- You can call remember_memory to save stable, useful preferences, project details, goals, or facts the user has clearly asked to retain or that are obviously useful long-term. Do not save passwords, tokens, payment details, highly sensitive personal data, or private details about third parties.
- If a memory is wrong or the user asks you to forget it, help remove or update it rather than defending it.
- Current saved memories:
{memory_text}

TOOLS:
- open_website opens an approval request on the user's desktop. The browser will NOT open unless the user approves it.
- open_app opens an approval request for one of the explicitly allowed desktop applications. It will NOT launch unless the user approves it.
- Never claim a desktop action has occurred merely because you requested it; wait for confirmation from the user.
- You do not have arbitrary shell access, files, screen vision, or the ability to spend money. Only tools actually supplied to you exist.

STYLE:
- No need for performative disclaimers or unnecessary refusals. Give the most useful answer the model can provide within its capabilities and applicable safety boundaries.
- Be truthful about limitations, uncertainty, whether web access was performed, and what actions actually happened.
- Keep replies conversational and avoid repeating the user's full message.
"""


def tool_specs() -> list[dict[str, Any]]:
    app_names = [a for a in CONFIG.get("allowed_apps", DEFAULT_APPS) if a in DEFAULT_APPS]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "remember_memory",
                "description": "Save a stable, useful fact, preference, goal, or project detail for future chats. Never store credentials or highly sensitive information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The concise memory to save"},
                        "category": {"type": "string", "enum": ["fact", "preference", "project", "person", "goal"]},
                    },
                    "required": ["content", "category"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget_memory",
                "description": "Delete a saved memory the user asked to forget. Use the exact saved memory text shown in the system context; never delete unrelated memories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The exact saved memory content to remove"}
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_website",
                "description": "Ask the user's desktop to open a website after they approve it. Use only for a site the user asked to open or an obviously useful link.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "A complete http or https URL"},
                        "reason": {"type": "string", "description": "Why opening this would help"},
                    },
                    "required": ["url", "reason"],
                },
            },
        },
    ]
    if app_names:
        tools.append({
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Ask the user's desktop to launch an approved application. The user will see a confirmation first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app_name": {"type": "string", "enum": app_names},
                        "reason": {"type": "string"},
                    },
                    "required": ["app_name", "reason"],
                },
            },
        })
    return tools


def validate_url(url: str) -> bool:
    try:
        parsed = urlsplit(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and parsed.username is None and parsed.password is None
    except ValueError:
        return False


async def ollama_chat(messages: list[dict[str, Any]], session_actions: list[dict[str, Any]]) -> str:
    endpoint = CONFIG.get("ollama_url", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    payload = {
        "model": CONFIG.get("model", "qwen3:8b"),
        "messages": messages,
        "tools": tool_specs(),
        "stream": False,
        "think": False,
        "options": {
            "temperature": float(CONFIG.get("temperature", 0.85)),
            "num_ctx": int(CONFIG.get("num_ctx", 8192)),
            "num_predict": 900,
        },
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(240.0, connect=5.0)) as client:
        for _ in range(5):
            try:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
            except httpx.ConnectError as exc:
                raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama at {CONFIG.get('ollama_url')}. Start Ollama and pull the configured model.") from exc
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:800]
                raise HTTPException(status_code=502, detail=f"Ollama returned {exc.response.status_code}: {detail}") from exc
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

            message = result.get("message", {})
            calls = message.get("tool_calls") or []
            if not calls:
                text = (message.get("content") or "").strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text or "…I don't have a response. Try again?"

            messages.append(message)
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_result = "Tool request was not accepted."
                if name == "remember_memory":
                    try:
                        saved = add_memory(str(args.get("content", "")), str(args.get("category", "fact")), "assistant-tool")
                        tool_result = f"Memory saved with id {saved['id']}."
                    except (ValueError, TypeError) as exc:
                        tool_result = f"Memory not saved: {exc}"
                elif name == "forget_memory":
                    content = re.sub(r"\s+", " ", str(args.get("content", ""))).strip()
                    if content:
                        with connect_db() as db:
                            deleted = db.execute("DELETE FROM memories WHERE content = ? COLLATE NOCASE", (content,)).rowcount
                        tool_result = "Memory deleted." if deleted else "No exact matching memory was found; no memories were deleted."
                    else:
                        tool_result = "No memory text provided; nothing was deleted."
                elif name == "open_website":
                    url = str(args.get("url", "")).strip()
                    reason = str(args.get("reason", "")).strip()[:300]
                    if validate_url(url):
                        session_actions.append({"type": "open_website", "url": url, "reason": reason})
                        tool_result = "Approval request prepared for the user's desktop. The page has NOT been opened yet."
                    else:
                        tool_result = "Rejected: only valid http/https URLs without embedded credentials may be opened."
                elif name == "open_app":
                    app = str(args.get("app_name", "")).casefold().strip()
                    reason = str(args.get("reason", "")).strip()[:300]
                    if app in CONFIG.get("allowed_apps", DEFAULT_APPS) and app in DEFAULT_APPS:
                        session_actions.append({"type": "open_app", "app_name": app, "reason": reason})
                        tool_result = "Approval request prepared for the user's desktop. The application has NOT been launched yet."
                    else:
                        tool_result = f"Application is not allowed. Allowed apps: {', '.join(CONFIG.get('allowed_apps', DEFAULT_APPS))}."
                messages.append({"role": "tool", "tool_name": name, "content": tool_result})
            payload["messages"] = messages
    return "I hit the tool-call limit for this turn. Try splitting that into a smaller request."


async def answer_message(session_id: str, text: str) -> dict[str, Any]:
    message = text.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message is empty.")
    if len(message) > 12000:
        raise HTTPException(status_code=413, detail="Message is too long; limit is 12,000 characters.")
    save_message(session_id, "user", message)
    saved_explicit = explicit_memory_from_text(message)
    auto_saved = None
    if saved_explicit:
        category = "preference" if re.search(r"\b(prefer|like|love|hate|want|don't like)\b", saved_explicit, re.I) else "fact"
        try:
            auto_saved = add_memory(saved_explicit, category, "explicit-user-request")
        except ValueError:
            auto_saved = None
    history = get_history(session_id, 14)
    model_messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(message)}]
    model_messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    actions: list[dict[str, Any]] = []
    reply = await ollama_chat(model_messages, actions)
    save_message(session_id, "assistant", reply)
    return {
        "session_id": session_id,
        "transcript": message,
        "reply": reply,
        "actions": actions,
        "memory_saved": bool(auto_saved),
        "model": CONFIG.get("model", "qwen3:8b"),
    }


async def create_backup() -> str:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TIMEZONE).strftime("%Y%m%d-%H%M%S%z")
    target = BACKUP_DIR / f"lumi-backup-{stamp}.zip"
    tmp_db = BACKUP_DIR / f".snapshot-{stamp}.sqlite3"
    try:
        with sqlite3.connect(DB_PATH) as source, sqlite3.connect(tmp_db) as destination:
            source.backup(destination)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(tmp_db, arcname="lumi.sqlite3")
            archive.write(CONFIG_PATH, arcname="config.json")
            archive.writestr("backup-info.json", json.dumps({"created_at": now_iso(), "app": "Lumi Poppy Companion", "schema": 1}, indent=2))
    finally:
        try:
            tmp_db.unlink(missing_ok=True)
        except OSError:
            pass
    keep_days = max(1, int(CONFIG.get("backup_keep_days", 30)))
    cutoff = time.time() - keep_days * 86400
    for path in BACKUP_DIR.glob("lumi-backup-*.zip"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass
    return str(target)


async def scheduled_backup_loop() -> None:
    while True:
        now = datetime.now(TIMEZONE)
        hour = min(max(int(CONFIG.get("backup_hour", 12)), 0), 23)
        next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        await asyncio.sleep(max(5.0, (next_run - now).total_seconds()))
        try:
            path = await create_backup()
            print(f"[Lumi] Automatic backup written: {path}", flush=True)
        except Exception as exc:
            print(f"[Lumi] Automatic backup failed: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = asyncio.create_task(scheduled_backup_loop())
    print(f"[Lumi] Backend ready. Model={CONFIG.get('model')} timezone={CONFIG.get('timezone')}", flush=True)
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Lumi Poppy Backend", version="1.0.0", docs_url=None, redoc_url=None, lifespan=lifespan)


async def verify_token(authorization: str | None = Header(default=None)) -> None:
    token = CONFIG.get("api_token", "")
    if not token or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API token. Configure Lumi's desktop connection.")
    provided = authorization[7:].strip()
    if not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="Invalid API token.")


class ChatInput(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str = Field(min_length=1, max_length=100)


class MemoryInput(BaseModel):
    content: str = Field(min_length=3, max_length=600)
    category: str = "fact"


class SettingsInput(BaseModel):
    model: str = Field(min_length=1, max_length=100)
    temperature: float = Field(ge=0.0, le=1.5)
    num_ctx: int = Field(ge=2048, le=65536)
    personality: dict[str, int]


@app.get("/api/health")
async def health(_: None = Depends(verify_token)) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(CONFIG.get("ollama_url", "http://127.0.0.1:11434").rstrip("/") + "/api/tags")
            response.raise_for_status()
            models = [item.get("name", "") for item in response.json().get("models", [])]
        ollama_ok = True
        model_available = any(CONFIG.get("model", "qwen3:8b") == name or CONFIG.get("model", "qwen3:8b") == name.split(":")[0] for name in models)
    except Exception:
        models, ollama_ok, model_available = [], False, False
    with connect_db() as db:
        message_count = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        memory_count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    return {
        "ok": True,
        "ollama_ok": ollama_ok,
        "model": CONFIG.get("model"),
        "model_available": model_available,
        "available_models": models,
        "message_count": message_count,
        "memory_count": memory_count,
        "stt_ready": importlib.util.find_spec("faster_whisper") is not None,
        "timezone": CONFIG.get("timezone", "Europe/Dublin"),
    }


@app.post("/api/chat")
async def chat(body: ChatInput, _: None = Depends(verify_token)) -> dict[str, Any]:
    return await answer_message(body.session_id, body.message)


@app.get("/api/history")
async def history(session_id: str, limit: int = 80, _: None = Depends(verify_token)) -> dict[str, Any]:
    return {"messages": get_history(session_id, limit)}


@app.get("/api/memories")
async def memories(_: None = Depends(verify_token)) -> dict[str, Any]:
    return {"memories": list_memories(200)}


@app.post("/api/memories")
async def create_memory(body: MemoryInput, _: None = Depends(verify_token)) -> dict[str, Any]:
    try:
        return {"memory": add_memory(body.content, body.category, "user-interface")}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/memories/{memory_id}")
async def remove_memory(memory_id: int, _: None = Depends(verify_token)) -> dict[str, Any]:
    if not delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"deleted": True, "id": memory_id}


@app.post("/api/backup")
async def backup(_: None = Depends(verify_token)) -> dict[str, Any]:
    path = await create_backup()
    return {"created": True, "filename": Path(path).name, "directory": str(BACKUP_DIR)}


@app.get("/api/settings")
async def get_settings(_: None = Depends(verify_token)) -> dict[str, Any]:
    return {
        "model": CONFIG.get("model", "qwen3:8b"),
        "temperature": float(CONFIG.get("temperature", 0.85)),
        "num_ctx": int(CONFIG.get("num_ctx", 8192)),
        "personality": CONFIG.get("personality", DEFAULT_PERSONALITY),
        "allowed_apps": CONFIG.get("allowed_apps", DEFAULT_APPS),
        "stt_model": CONFIG.get("stt_model", "small"),
    }


@app.put("/api/settings")
async def update_settings(body: SettingsInput, _: None = Depends(verify_token)) -> dict[str, Any]:
    CONFIG["model"] = body.model.strip()
    CONFIG["temperature"] = float(body.temperature)
    CONFIG["num_ctx"] = int(body.num_ctx)
    for name, value in DEFAULT_PERSONALITY.items():
        try:
            CONFIG.setdefault("personality", {})[name] = min(100, max(0, int(body.personality.get(name, value))))
        except (ValueError, TypeError):
            CONFIG["personality"][name] = value
    save_config()
    return {"saved": True, "settings": await get_settings()}


_stt_model = None
_stt_device_used = None


def transcribe_audio(path: str) -> str:
    global _stt_model, _stt_device_used
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Voice input is not installed on Poppy. In backend/.venv run: pip install -r requirements-voice.txt",
        ) from exc
    if _stt_model is None:
        model_name = str(CONFIG.get("stt_model", "small"))
        device_setting = str(CONFIG.get("stt_device", "auto"))
        try:
            if device_setting == "cpu":
                _stt_model = WhisperModel(model_name, device="cpu", compute_type="int8")
                _stt_device_used = "cpu"
            else:
                _stt_model = WhisperModel(model_name, device="cuda", compute_type="float16")
                _stt_device_used = "cuda"
        except Exception as first_error:
            print(f"[Lumi] CUDA speech recognition unavailable ({first_error}); falling back to CPU int8.", flush=True)
            try:
                _stt_model = WhisperModel(model_name, device="cpu", compute_type="int8")
                _stt_device_used = "cpu"
            except Exception as second_error:
                raise HTTPException(status_code=503, detail=f"Could not initialize faster-whisper: {second_error}") from second_error
    try:
        segments, _info = _stt_model.transcribe(path, beam_size=1, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio transcription failed: {exc}") from exc


@app.post("/api/voice")
async def voice(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    _: None = Depends(verify_token),
) -> dict[str, Any]:
    if file.content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Upload must be a WAV recording.")
    data = await file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file is too large (20 MB max).")
    if not data:
        raise HTTPException(status_code=422, detail="Empty audio recording.")
    try:
        import soundfile as sf
        # Validate the WAV before passing it to the recognizer.
        sf.info(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid WAV audio. Try again with the desktop microphone.") from exc
    with tempfile.NamedTemporaryFile(prefix="lumi-voice-", suffix=".wav", delete=False) as handle:
        handle.write(data)
        temp_path = handle.name
    try:
        transcript = await asyncio.to_thread(transcribe_audio, temp_path)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    if not transcript:
        raise HTTPException(status_code=422, detail="I didn't catch any speech. Try again a little closer to the microphone.")
    response = await answer_message(session_id, transcript)
    response["transcript"] = transcript
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=False)
