import os
import re
import time
import subprocess
import uuid
import requests
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
WORK_DIR = "/tmp/clips"
os.makedirs(WORK_DIR, exist_ok=True)

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_BASE = "https://api.assemblyai.com/v2"


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Clip pipeline service is running"})


def to_direct_download_link(url):
    """Converts common Google Drive / Dropbox share links into direct-download links."""
    drive_match = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if drive_match:
        file_id = drive_match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    if "drive.google.com/open?id=" in url:
        file_id = url.split("id=")[-1].split("&")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    if "dropbox.com" in url:
        if "?dl=0" in url:
            return url.replace("?dl=0", "?dl=1")
        if "?dl=1" not in url:
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}dl=1"
        return url

    return url  # assume it's already a direct file link


def download_file(video_url, job_id):
    """Downloads a video file directly (Google Drive, Dropbox, or any direct link)."""
    direct_url = to_direct_download_link(video_url)
    out_path = os.path.join(WORK_DIR, f"{job_id}_source.mp4")

    with requests.get(direct_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return out_path


def extract_audio(video_path, job_id):
    """Pulls audio out of a downloaded video file as mp3, for transcription."""
    audio_path = os.path.join(WORK_DIR, f"{job_id}_audio.mp3")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", audio_path]
    subprocess.run(cmd, check=True, timeout=300)
    return audio_path


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Body: { "video_url": "https://..." }
    Downloads audio, sends to AssemblyAI, waits for result, returns
    the full transcript text plus word-level timestamps (in ms).
    """
    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not set on server"}), 500

    data = request.get_json(force=True)
    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "video_url is required"}), 400

    job_id = str(uuid.uuid4())
    video_path = None
    audio_path = None

    try:
        video_path = download_file(video_url, job_id)
        audio_path = extract_audio(video_path, job_id)

        headers = {"authorization": ASSEMBLYAI_API_KEY}

        # Upload audio file to AssemblyAI
        with open(audio_path, "rb") as f:
            upload_resp = requests.post(
                f"{ASSEMBLYAI_BASE}/upload", headers=headers, data=f
            )
        upload_resp.raise_for_status()
        audio_url = upload_resp.json()["upload_url"]

        # Request transcription
        transcript_resp = requests.post(
            f"{ASSEMBLYAI_BASE}/transcript",
            headers=headers,
            json={"audio_url": audio_url},
        )
        transcript_resp.raise_for_status()
        transcript_id = transcript_resp.json()["id"]

        # Poll until done
        polling_url = f"{ASSEMBLYAI_BASE}/transcript/{transcript_id}"
        while True:
            poll_resp = requests.get(polling_url, headers=headers)
            poll_resp.raise_for_status()
            result = poll_resp.json()
            if result["status"] == "completed":
                break
            elif result["status"] == "error":
                return jsonify({"error": f"Transcription failed: {result.get('error')}"}), 500
            time.sleep(3)

        return jsonify({
            "text": result.get("text", ""),
            "words": result.get("words", []),  # each: {text, start, end} in ms
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Download failed: {str(e)}"}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"AssemblyAI request failed: {str(e)}"}), 500
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def ms_to_srt_time(ms):
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def words_to_srt(words, clip_start_ms, chunk_size=6):
    """Groups words into caption lines and writes an SRT file, with
    timestamps shifted so 0 = start of the clip (not the full video)."""
    lines = []
    idx = 1
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue
        start = chunk[0]["start"] - clip_start_ms
        end = chunk[-1]["end"] - clip_start_ms
        if end < 0:
            continue
        start = max(start, 0)
        text = " ".join(w["text"] for w in chunk)
        lines.append(f"{idx}\n{ms_to_srt_time(start)} --> {ms_to_srt_time(end)}\n{text}\n")
        idx += 1
    return "\n".join(lines)


@app.route("/clip_with_captions", methods=["POST"])
def clip_with_captions():
    """
    Body:
    {
      "video_url": "https://...",
      "start_ms": 83000,
      "end_ms": 113000,
      "words": [ {"text": "hello", "start": 83200, "end": 83500}, ... ]
    }
    "words" should be the slice of the full transcript's words that fall
    within [start_ms, end_ms] (Make can filter these before calling this).
    Cuts the clip and burns in captions. Returns the finished MP4.
    """
    data = request.get_json(force=True)
    video_url = data.get("video_url")
    start_ms = data.get("start_ms")
    end_ms = data.get("end_ms")
    words = data.get("words", [])

    if video_url is None or start_ms is None or end_ms is None:
        return jsonify({"error": "video_url, start_ms, and end_ms are required"}), 400

    job_id = str(uuid.uuid4())
    source_path = None
    srt_path = os.path.join(WORK_DIR, f"{job_id}.srt")
    clip_path = os.path.join(WORK_DIR, f"{job_id}_clip.mp4")

    def ms_to_ts(ms):
        s = ms / 1000
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02}:{m:02}:{sec:06.3f}"

    try:
        source_path = download_file(video_url, job_id)

        # Write SRT file for this clip's caption window
        srt_content = words_to_srt(words, start_ms)
        with open(srt_path, "w") as f:
            f.write(srt_content)

        start_ts = ms_to_ts(start_ms)
        end_ts = ms_to_ts(end_ms)

        # Cut + burn captions in one ffmpeg pass
        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-ss", start_ts,
            "-to", end_ts,
            "-vf", f"subtitles={srt_path}:force_style='FontSize=22,Bold=1'",
            "-c:v", "libx264",
            "-c:a", "aac",
            clip_path,
        ]
        subprocess.run(cmd, check=True, timeout=600)

        return send_file(clip_path, as_attachment=True, download_name="clip.mp4")

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        if source_path and os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(srt_path):
            os.remove(srt_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
            
