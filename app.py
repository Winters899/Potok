from flask import Flask, render_template, request, send_file, jsonify
import edge_tts
import asyncio
import os
import uuid
from bs4 import BeautifulSoup
from pypdf import PdfReader

app = Flask(__name__)

AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)

ALLOWED_VOICES = {
    "ru-RU-DmitryNeural",
    "ru-RU-SvetlanaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
}

MAX_TEXT_LEN = 1_000_000

# === ОДИН EVENT LOOP НА ВСЁ ПРИЛОЖЕНИЕ ===
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract_text", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    text = ""

    try:
        if file.filename.lower().endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

        elif file.filename.lower().endswith(".fb2"):
            soup = BeautifulSoup(file.read(), "xml")
            text = "\n".join(p.get_text() for p in soup.find_all("p"))

        else:
            return jsonify({"error": "Unsupported format"}), 400

        return jsonify({"text": text.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(force=True)
    text = data.get("text", "")
    voice = data.get("voice", "ru-RU-DmitryNeural")

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400
    if len(text) > MAX_TEXT_LEN:
        return jsonify({"error": "Text too long"}), 413
    if voice not in ALLOWED_VOICES:
        return jsonify({"error": "Unsupported voice"}), 400

    file_id = uuid.uuid4().hex
    audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")

    try:
        marks = loop.run_until_complete(
            generate_with_timings(text, voice, audio_path)
        )

        return jsonify({
            "audio_url": f"/get_audio/{file_id}.wav",
            "marks": marks
        })

    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({"error": str(e)}), 500


@app.route("/get_audio/<filename>")
def get_audio(filename):
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="audio/wav")


# === EDGE TTS С МЕТКАМИ ===
async def generate_with_timings(text, voice, filepath):
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    marks = []

    try:
        with open(filepath, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])

                elif chunk["type"] == "WordBoundary":
                    marks.append({
                        "offset": chunk["offset"] / 10_000_000,
                        "text_offset": chunk["text_offset"],
                        "word_len": chunk["word_length"],
                        "word": text[
                            chunk["text_offset"]:
                            chunk["text_offset"] + chunk["word_length"]
                        ]
                    })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise e

    return marks

if __name__ == "__main__":
    app.run(debug=True, port=5000)