from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import edge_tts
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, send_from_directory
from pypdf import PdfReader

app = Flask(__name__)

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "audio_cache"
# Гарантируем, что каталог кэша существует до любых операций записи
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VOICES = {
    "ru-RU-DmitryNeural",
    "ru-RU-SvetlanaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
}

MAX_TEXT_LEN = 1_000_000

AUDIO_EXT = ".mp3"
AUDIO_MIME = "audio/mpeg"
MARKS_EXT = ".json"

# Ограничим параллельные генерации (чтобы не уронить edge-tts/канал)
MAX_CONCURRENT_SYNTH = 2
synth_sem = asyncio.Semaphore(MAX_CONCURRENT_SYNTH)

# Очистка кэша
CACHE_MAX_AGE_SEC = 3 * 24 * 3600
CACHE_MAX_PAIRS = 2000
CACHE_CLEANUP_EVERY_SEC = 10 * 60

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("potok")

# ----------------------------
# Helpers
# ----------------------------
def _safe_voice(v: str) -> str:
    return v if v in ALLOWED_VOICES else "ru-RU-DmitryNeural"


def _cache_key(text: str, voice: str) -> str:
    h = hashlib.sha256()
    h.update(voice.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _audio_name(key: str) -> str:
    return f"{key}{AUDIO_EXT}"


def _marks_name(key: str) -> str:
    return f"{key}{MARKS_EXT}"


def _audio_path(key: str) -> Path:
    return CACHE_DIR / _audio_name(key)


def _marks_path(key: str) -> Path:
    return CACHE_DIR / _marks_name(key)


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)  # atomic replace [web:132]


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_cache_once() -> None:
    now = time.time()

    audio_files = {p.stem: p for p in CACHE_DIR.glob(f"*{AUDIO_EXT}") if p.is_file()}
    marks_files = {p.stem: p for p in CACHE_DIR.glob(f"*{MARKS_EXT}") if p.is_file()}
    keys = set(audio_files.keys()) | set(marks_files.keys())

    removed = 0

    # 1) удалить сироты (есть только audio или только marks)
    for k in list(keys):
        a = audio_files.get(k)
        m = marks_files.get(k)
        if (a is None) != (m is None):
            try:
                if a and a.exists():
                    a.unlink(missing_ok=True)
                    removed += 1
                if m and m.exists():
                    m.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass

    # 2) собрать пары и удалить старые/лишние
    pairs: List[tuple[str, float]] = []
    for a in CACHE_DIR.glob(f"*{AUDIO_EXT}"):
        m = CACHE_DIR / (a.stem + MARKS_EXT)
        if not m.exists():
            continue
        try:
            mt = max(a.stat().st_mtime, m.stat().st_mtime)
            pairs.append((a.stem, mt))
        except FileNotFoundError:
            continue

    pairs.sort(key=lambda x: x[1])  # старые первыми

    # 2a) по возрасту
    for k, mt in pairs:
        if now - mt <= CACHE_MAX_AGE_SEC:
            continue
        try:
            _audio_path(k).unlink(missing_ok=True)
            _marks_path(k).unlink(missing_ok=True)
            removed += 2
        except OSError:
            pass

    # 2b) по количеству
    pairs = []
    for a in CACHE_DIR.glob(f"*{AUDIO_EXT}"):
        m = CACHE_DIR / (a.stem + MARKS_EXT)
        if not m.exists():
            continue
        try:
            mt = max(a.stat().st_mtime, m.stat().st_mtime)
            pairs.append((a.stem, mt))
        except FileNotFoundError:
            continue

    pairs.sort(key=lambda x: x[1])
    overflow = max(0, len(pairs) - CACHE_MAX_PAIRS)
    for i in range(overflow):
        k = pairs[i][0]
        try:
            _audio_path(k).unlink(missing_ok=True)
            _marks_path(k).unlink(missing_ok=True)
            removed += 2
        except OSError:
            pass

    if removed:
        log.info("Cache cleanup removed=%s", removed)


_cleanup_thread_started = False

def start_cache_cleanup_thread_once() -> None:
    global _cleanup_thread_started
    if _cleanup_thread_started:
        return
    _cleanup_thread_started = True

    def worker() -> None:
        while True:
            time.sleep(CACHE_CLEANUP_EVERY_SEC)
            try:
                _cleanup_cache_once()
            except Exception:
                log.exception("Cache cleanup thread error")

    t = threading.Thread(target=worker, name="cache-cleaner", daemon=True)
    t.start()
    log.info("Cache cleanup thread started")


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/extract_text")
def extract_text():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            parts: List[str] = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
            text = "\n".join(parts).strip()

        elif filename.endswith(".fb2"):
            soup = BeautifulSoup(file.read(), "xml")
            text = "\n".join(p.get_text() for p in soup.find_all("p")).strip()

        else:
            return jsonify({"error": "Unsupported format"}), 400

        if len(text) > MAX_TEXT_LEN:
            return jsonify({"error": "Extracted text too long"}), 413

        return jsonify({"text": text})
    except Exception as e:
        log.exception("extract_text failed")
        return jsonify({"error": str(e)}), 500


@app.post("/speak")
async def speak():
    # async view поддерживается Flask при установке Flask[async] [web:121]
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = _safe_voice(data.get("voice") or "ru-RU-DmitryNeural")
    
    preview = text[:80].replace("\n", "\\n")
    tail = text[-80:].replace("\n", "\\n")
    log.info("speak text preview=%r tail=%r", preview, tail)


    if not text:
        return jsonify({"error": "Empty text"}), 400
    if len(text) > MAX_TEXT_LEN:
        return jsonify({"error": "Text too long"}), 413

    key = _cache_key(text, voice)
    a_path = _audio_path(key)
    m_path = _marks_path(key)

    # Cache hit
    if a_path.exists() and m_path.exists():
        try:
            marks = await asyncio.to_thread(_read_json, m_path)
            log.info("speak cache hit key=%s voice=%s marks=%s", key[:12], voice, len(marks))
            return jsonify({
                "audio_url": f"/get_audio/{_audio_name(key)}",
                "marks": marks,
                "cache": True,
            })
        except Exception:
            # битый json -> удаляем пару и перегенерим
            a_path.unlink(missing_ok=True)
            m_path.unlink(missing_ok=True)

    # Неконсистентный кэш -> удалить
    if a_path.exists() != m_path.exists():
        a_path.unlink(missing_ok=True)
        m_path.unlink(missing_ok=True)

    log.info("speak generate key=%s voice=%s len=%s", key[:12], voice, len(text))

    try:
        async with synth_sem:
            marks = await generate_with_timings(text=text, voice=voice, audio_path=a_path)

        await asyncio.to_thread(_atomic_write_json, m_path, marks)  # atomic [web:132]

        return jsonify({
            "audio_url": f"/get_audio/{_audio_name(key)}",
            "marks": marks,
            "cache": False,
        })
    except Exception as e:
        a_path.unlink(missing_ok=True)
        m_path.unlink(missing_ok=True)
        log.exception("speak failed key=%s", key[:12])
        return jsonify({"error": str(e)}), 500


@app.get("/get_audio/<path:filename>")
def get_audio(filename: str):
    if not filename.endswith(AUDIO_EXT):
        return "Not found", 404

    path = CACHE_DIR / filename
    if not path.exists():
        return "Not found", 404

    return send_from_directory(CACHE_DIR, filename, mimetype=AUDIO_MIME, conditional=True)


# ----------------------------
# edge-tts
# ----------------------------
async def generate_with_timings(*, text: str, voice: str, audio_path: Path) -> List[Dict[str, Any]]:
    """
    WordBoundary offset обычно в 100-нс "тиках"; перевод в секунды: / 10_000_000. [web:54]
    """
    communicate = edge_tts.Communicate(text, voice)
    marks: List[Dict[str, Any]] = []

    # путь к временному файлу рядом с целевым mp3
    tmp_audio = audio_path.with_suffix(audio_path.suffix + ".part")

    # на всякий случай убеждаемся, что директория существует
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(tmp_audio, "wb") as f:
            async for chunk in communicate.stream():
                tp = chunk.get("type")

                if tp == "audio":
                    f.write(chunk["data"])

                elif tp == "WordBoundary":
                    text_offset = int(chunk["text_offset"])
                    word_len = int(chunk["word_length"])

                    marks.append({
                        "offset": float(chunk["offset"]) / 10_000_000,  # seconds [web:54]
                        "text_offset": text_offset,
                        "word_len": word_len,
                        "word": text[text_offset:text_offset + word_len].strip(),
                    })

        os.replace(tmp_audio, audio_path)  # atomic [web:132]
        return marks

    except Exception:
        try:
            tmp_audio.unlink(missing_ok=True)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    # Flask рекомендует делать setup до app.run() (а не через несуществующие lifecycle hooks) [web:170]
    _cleanup_cache_once()
    start_cache_cleanup_thread_once()

    # reloader лучше выключить, чтобы не запускать дважды
    app.run(debug=True, port=5000, use_reloader=False)
