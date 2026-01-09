from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
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
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = BASE_DIR / "progress.json"

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
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_UNI_GLYPH_RE = re.compile(r"/?uni([0-9A-Fa-f]{4})")


def _decode_uni_glyph_names(s: str) -> str:
    """
    pypdf иногда возвращает glyph names вида "/uni041A/uni0438..." вместо текста.
    Пробуем превратить uniXXXX обратно в символы Unicode.
    """
    if not s:
        return ""

    replaced = 0

    def repl(m: re.Match) -> str:
        nonlocal replaced
        replaced += 1
        return chr(int(m.group(1), 16))

    out = _UNI_GLYPH_RE.sub(repl, s)
    if replaced:
        log.info("pdf_clean: decoded uniXXXX=%s", replaced)

    # После подстановки часто остаются "/" между глифами — убираем,
    # но только когда слеш стоит между буквами/цифрами (не трогаем URL/пути).
    out = re.sub(r"(?<=\w)/(?=\w)", "", out)
    return out


def _clean_pdf_text(s: str) -> str:
    """
    Очистка текста после извлечения из PDF:
    - декод /uniXXXX
    - NBSP -> пробел
    - лидеры оглавления ". . . . ." -> " — "
    - убираем soft hyphen
    - склейка переносов слов
    - одиночные переносы строк -> пробел
    - схлопывание множественных пробелов
    """
    if not s:
        return ""

    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Декодируем /uniXXXX (если встретились)
    s = _decode_uni_glyph_names(s)

    # NBSP -> обычный пробел (U+00A0 часто используется в PDF)
    s = s.replace("\xa0", " ")

    # Оглавление: ".  .  .  .  ." (dot leaders) -> " — "
    # (иначе TTS проговаривает точки)
    # Лидеры оглавления -> " — " только если строка заканчивается номером страницы
    # Пример: "Глава 5 ... 186"  => "Глава 5 — 186"
    s = re.sub(
        r"(?:\s*\.\s*){4,}(?=\s*\d+\s*$)",
        " — ",
        s,
        flags=re.MULTILINE,
    )


    # Часто в PDF встречается "мягкий перенос" (soft hyphen), его TTS читает как дефис
    s = s.replace("\u00ad", "")

    # Нормализуем разные виды дефиса/минуса к обычному "-"
    s = s.translate(
        {
            ord("\u2010"): ord("-"),  # hyphen
            ord("\u2011"): ord("-"),  # non-breaking hyphen
            ord("\u2212"): ord("-"),  # minus sign
        }
    )

    # Склеиваем переносы слов: \w - whitespace/ZW - \w
    s = re.sub(r"(?<=\w)-[\s\u200b\u2060]+(?=\w)", "", s)

    # Одиночные переносы строк внутри абзаца заменяем на пробел (двойные \n оставляем)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)

    # Нормализуем множественные пробелы
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()


def _log_pdf_hyphen_issues(*, label: str, s: str, limit: int = 12) -> None:
    """
    Диагностика: логируем фрагменты, где видно дефис + пробел(ы) между \w...\w
    и спец-символы переносов.
    """
    if not s:
        return

    specials = {
        "\u00ad": "SOFT_HYPHEN",
        "\u2010": "HYPHEN",
        "\u2011": "NON_BREAKING_HYPHEN",
        "\u2212": "MINUS",
        "\u200b": "ZERO_WIDTH_SPACE",
        "\u2060": "WORD_JOINER",
    }
    found = {name: s.count(ch) for ch, name in specials.items() if ch in s}
    if found:
        log.info("pdf_clean[%s] specials=%s", label, found)

    rx = re.compile(r"(?<=\w)-\s+(?=\w)")
    hits = []
    for m in rx.finditer(s):
        a = max(0, m.start() - 20)
        b = min(len(s), m.end() + 20)
        hits.append((m.start(), s[a:b]))
        if len(hits) >= limit:
            break
    if hits:
        log.info(
            "pdf_clean[%s] hyphen_whitespace_hits=%s sample=%r",
            label,
            len(hits),
            hits[: min(3, len(hits))],
        )


def _log_pdf_unicode_glyphs(*, label: str, page_index: int, s: str) -> None:
    """
    Диагностика кейса, когда текст выглядит как "/uni041A/uni0438..."
    """
    if not s:
        return
    n = len(re.findall(r"uni[0-9A-Fa-f]{4}", s))
    if n:
        log.info(
            "pdf_text[%s] page=%s uniXXXX_count=%s preview=%r",
            label,
            page_index,
            n,
            s[:120],
        )


# ----------------------------
# Chapter detection
# ----------------------------
def _detect_chapters_fb2(root: BeautifulSoup) -> list[dict]:
    """
    Меню глав по структуре FB2: body/section/title.
    """
    chapters: list[dict] = []

    def walk_section(section, level: int) -> None:
        title_tag = section.find("title", recursive=False)
        if title_tag:
            title_text = " ".join(p.get_text(strip=True) for p in title_tag.find_all("p"))
            if title_text:
                chapters.append({"title": title_text, "level": level})

        for child in section.find_all("section", recursive=False):
            walk_section(child, level + 1)

    for body in root.find_all("body"):
        body_name = body.get("name", "")
        if body_name == "notes":
            continue

        for section in body.find_all("section", recursive=False):
            walk_section(section, level=1)

    return chapters


def _detect_chapters_pdf_outlines(reader: PdfReader) -> list[dict]:
    """
    Главы из закладок (outlines/bookmarks).
    """
    chapters: list[dict] = []
    try:
        outlines = reader.outline
    except Exception:
        try:
            outlines = reader.outlines
        except Exception:
            outlines = []

    def walk(items, level: int) -> None:
        from pypdf.generic import Destination

        for it in items:
            if isinstance(it, list):
                walk(it, level + 1)
                continue
            try:
                if isinstance(it, Destination):
                    title = str(it.title)
                    page_num = reader.get_destination_page_number(it)
                else:
                    title = str(getattr(it, "title", "") or it)
                    dest = reader.get_destination(it)
                    page_num = reader.get_destination_page_number(dest)

                if page_num is None:
                    continue

                chapters.append({"title": title.strip(), "level": level, "page": int(page_num)})
            except Exception:
                continue

    try:
        if outlines:
            walk(outlines, level=1)
    except Exception:
        chapters = []

    seen = set()
    result: list[dict] = []
    for ch in chapters:
        key = (ch.get("title"), ch.get("page"))
        if not ch.get("title") or key in seen:
            continue
        seen.add(key)
        result.append(ch)
    return result


def _detect_chapters_pdf_text(pages: list[str]) -> list[dict]:
    """
    Fallback для PDF: ищем заголовки глав в plain-text.
    """
    chapter_re = re.compile(
        r"^\s*(глава|часть|раздел|chapter|part)\s+\d+.*$|"
        r"^\s*(вступление|введение|заключение|об авторе|предисловие|послесловие|"
        r"introduction|preface|conclusion)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    chapters: list[dict] = []
    for i, page_text in enumerate(pages):
        for m in chapter_re.finditer(page_text):
            title = m.group(0).strip()
            chapters.append({"title": title, "level": 1, "page": i})

    seen = set()
    result: list[dict] = []
    for ch in chapters:
        key = (ch["title"], ch["page"])
        if key in seen:
            continue
        seen.add(key)
        result.append(ch)
    return result


def _detect_chapters_text_plain(text: str) -> list[dict]:
    """
    Fallback для FB2/любого текста: ищем заголовки по шаблонам.
    """
    patterns = [
        r"^\s*(глава|часть|раздел|chapter|part)\s+[IVXLCDM\d]+[:\.\s].*$",
        r"^\s*([IVXLCDM]+|\d+)\.\s+[А-ЯA-Z].*$",
        r"^\s*(вступление|введение|заключение|об авторе|предисловие|послесловие|эпилог|пролог|"
        r"introduction|preface|conclusion|epilogue|prologue)\s*$",
        r"^[А-ЯA-Z][А-ЯA-Z\s]{10,}$",
    ]

    combined = "|".join(f"({p})" for p in patterns)
    chapter_re = re.compile(combined, re.IGNORECASE | re.MULTILINE)

    chapters: list[dict] = []
    for m in chapter_re.finditer(text):
        title = m.group(0).strip()
        if len(title) > 200:
            continue
        start = m.start()
        chapters.append({"title": title, "level": 1, "start_char": start})

    seen = set()
    result: list[dict] = []
    for ch in chapters:
        key = (ch["title"], ch["start_char"])
        if key in seen:
            continue
        seen.add(key)
        result.append(ch)
    return result


# ----------------------------
# Cache cleanup
# ----------------------------
def _cleanup_cache_once() -> None:
    now = time.time()

    audio_files = {p.stem: p for p in CACHE_DIR.glob(f"*{AUDIO_EXT}") if p.is_file()}
    marks_files = {p.stem: p for p in CACHE_DIR.glob(f"*{MARKS_EXT}") if p.is_file()}
    keys = set(audio_files.keys()) | set(marks_files.keys())

    removed = 0

    # удалить одиночки (mp3 без json или наоборот)
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

    # собрать пары с mtime
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

    # чистка по возрасту
    pairs.sort(key=lambda x: x[1])
    for k, mt in pairs:
        if now - mt <= CACHE_MAX_AGE_SEC:
            continue
        try:
            _audio_path(k).unlink(missing_ok=True)
            _marks_path(k).unlink(missing_ok=True)
            removed += 2
        except OSError:
            pass

    # чистка по количеству
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
        chapters: list[dict] = []
        log.info("extract_text: filename=%r", filename)

        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            pages: List[str] = []

            for i, page in enumerate(reader.pages):
                t_raw = page.extract_text() or ""

                # Чтобы не засорять лог: детально логируем только первые 5 страниц
                if t_raw and i < 5:
                    _log_pdf_hyphen_issues(label="before", s=t_raw, limit=6)

                if t_raw:
                    _log_pdf_unicode_glyphs(label="raw", page_index=i, s=t_raw)

                t = _clean_pdf_text(t_raw)

                if t and i < 5:
                    _log_pdf_hyphen_issues(label="after", s=t, limit=6)

                if t:
                    _log_pdf_unicode_glyphs(label="clean", page_index=i, s=t)
                    pages.append(t)

            text = "\n\n".join(pages).strip()
            log.info("extract_text: pdf pages=%s text_len=%s", len(pages), len(text))
            log.info("extract_text: pdf cleaned preview=%r", text[:200])
            log.info("extract_text: pdf cleaned tail=%r", text[-200:])

            chapters = _detect_chapters_pdf_outlines(reader)
            log.info("extract_text: pdf outlines chapters=%s", len(chapters))
            if not chapters:
                chapters = _detect_chapters_pdf_text(pages)
                log.info("extract_text: pdf text chapters=%s", len(chapters))

        elif filename.endswith(".fb2"):
            raw = file.read()
            soup = BeautifulSoup(raw, "xml")
            text = "\n".join(p.get_text() for p in soup.find_all("p")).strip()

            chapters = _detect_chapters_fb2(soup)
            log.info("extract_text: fb2 structural chapters=%s", len(chapters))

            if not chapters:
                chapters = _detect_chapters_text_plain(text)
                log.info("extract_text: fb2 text-fallback chapters=%s", len(chapters))

        else:
            return jsonify({"error": "Unsupported format"}), 400

        if len(text) > MAX_TEXT_LEN:
            return jsonify({"error": "Extracted text too long"}), 413

        log.info("extract_text: ok text_len=%s chapters=%s", len(text), len(chapters))
        return jsonify({"text": text, "chapters": chapters})

    except Exception as e:
        log.exception("extract_text failed")
        return jsonify({"error": str(e)}), 500


@app.post("/save_progress")
def save_progress() -> tuple[str, int] | tuple[dict, int]:
    data = request.get_json(force=True, silent=True) or {}
    try:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400

        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log.info(
            "save_progress: len=%s voice=%s chunk_index=%s position=%.2f",
            len(text),
            data.get("voice"),
            data.get("chunk_index"),
            float(data.get("position") or 0.0),
        )
        return "", 204
    except Exception:
        log.exception("save_progress failed")
        return jsonify({"error": "save_progress failed"}), 500


@app.get("/load_progress")
def load_progress():
    if not PROGRESS_PATH.exists():
        return jsonify({})

    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        log.info(
            "load_progress: text_len=%s voice=%s chunk_index=%s position=%s",
            len(data.get("text") or ""),
            data.get("voice"),
            data.get("chunk_index"),
            data.get("position"),
        )
        return jsonify(data)
    except Exception:
        log.exception("load_progress failed")
        return jsonify({}), 500


@app.post("/speak")
def speak():
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

    if a_path.exists() and m_path.exists():
        try:
            marks = _read_json(m_path)
            log.info("speak cache hit key=%s voice=%s marks=%s", key[:12], voice, len(marks))
            return jsonify(
                {"audio_url": f"/get_audio/{_audio_name(key)}", "marks": marks, "cache": True}
            )
        except Exception:
            a_path.unlink(missing_ok=True)
            m_path.unlink(missing_ok=True)

    if a_path.exists() != m_path.exists():
        a_path.unlink(missing_ok=True)
        m_path.unlink(missing_ok=True)

    log.info("speak generate key=%s voice=%s len=%s", key[:12], voice, len(text))

    try:
        marks = asyncio.run(generate_with_timings(text=text, voice=voice, audio_path=a_path))
        _atomic_write_json(m_path, marks)
        return jsonify(
            {"audio_url": f"/get_audio/{_audio_name(key)}", "marks": marks, "cache": False}
        )
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
    WordBoundary offset в 100-нс тиках; перевод в секунды: / 10_000_000.
    """
    communicate = edge_tts.Communicate(text, voice)
    marks: List[Dict[str, Any]] = []

    tmp_audio = audio_path.with_suffix(audio_path.suffix + ".part")
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

                    marks.append(
                        {
                            "offset": float(chunk["offset"]) / 10_000_000,
                            "text_offset": text_offset,
                            "word_len": word_len,
                            "word": text[text_offset : text_offset + word_len].strip(),
                        }
                    )

        os.replace(tmp_audio, audio_path)
        return marks

    except Exception:
        try:
            tmp_audio.unlink(missing_ok=True)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    _cleanup_cache_once()
    start_cache_cleanup_thread_once()
    app.run(debug=True, port=5000, use_reloader=False)
