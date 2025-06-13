import sys
import whisper
import requests
import srt
import json
import openai
import jsonschema
import logging
import os
import argparse
from tqdm import tqdm
from datetime import timedelta
from pydub import AudioSegment
from moviepy import VideoFileClip
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("pysub.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress OpenAI client HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    schema_path = os.path.join(os.path.dirname(__file__), "schemas/pysub.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=config, schema=schema)
    return config

def extract_audio(video_path, audio_path="temp_audio.mp3"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

def translate_text(text, target_language, api_key=None, provider="openai", ollama_model="llama3"):
    if provider == "openai":
        return translate_with_openai(text, target_language, api_key)
    elif provider == "ollama":
        result = translate_with_ollama(text, target_language, model=ollama_model)
        return verify_or_retranslate_ollama(text, result, target_language, model=ollama_model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def translate_with_openai(text, target_language, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": (
                f"You are a professional translation engine. "
                f"Translate the following English sentence into **{target_language}** only. Do not use any other language except the one provide. "
                f"You must translate from english to **{target_language}**. Do not confuse one language with another. Check and verify your work after you are done"
                f"Respond with ONLY the *{target_language} sentence, no extra commentary."
            )
            },
            {
            "role": "user",
            "content": f"{text}"
            }
        ]
    )
    return response.choices[0].message.content.strip()

def translate_with_ollama(text, target_language, model="llama3"):
    prompt = (
        f"You are a highly accurate and reliable AI translator. "
        f"Your task is to translate the following English sentence into **{target_language}**.\n\n"

        f"🔒 Strict Output Rules:\n"
        f"1. The translation MUST be written exclusively in **{target_language} script**.\n"
        f"2. You MUST NOT include:\n"
        f"   - Romanized or transliterated text\n"
        f"   - English words or phrases\n"
        f"   - Commentary, notes, or metadata\n"
        f"   - The original English sentence\n"
        f"3. The output MUST be clear, fluent, and natural to a native speaker of {target_language}.\n"
        f"4. Your highest priority is accuracy and clarity for native speakers.\n"
        f"5. ✅ After translating, **you MUST internally verify** that the output contains only {target_language} script and NO foreign language content.\n"
        f"6. If verification fails, silently retry the translation until the output is clean and correct.\n\n"

        f"---\nInput:\n{text}\n\n"
        f"Output ({target_language} only):"
    )
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"].strip().strip('"')

def process_single_video(video_path, srt_path, config, parent_bar=None):
    translate = config.get("translate", False)
    language = config.get("target_language", "thai")
    api_key = config.get("api_key")
    provider = config.get("provider", "openai")
    ollama_model = config.get("ollama_model", "llama3")
    chunk_duration_sec = config.get("chunk_duration_sec", 60)
    chunk_overlap_sec = config.get("chunk_overlap_sec", 5)

    audio_path = extract_audio(video_path)
    audio = AudioSegment.from_file(audio_path)
    chunk_len = chunk_duration_sec * 1000
    overlap = chunk_overlap_sec * 1000
    chunks = [audio[i:i + chunk_len] for i in range(0, len(audio), chunk_len - overlap)]

    srt_index = 1
    last_english = None
    offset = 0.0

    with open(srt_path, "w", encoding="utf-8") as srt_file, tqdm(
        total=len(chunks),
        desc=f"Chunks [{os.path.basename(video_path)}]",
        position=1,
        leave=False
    ) as chunk_bar:

        for i, chunk in enumerate(chunks):
            chunk_file = f"chunk_{i}.mp3"
            chunk.export(chunk_file, format="mp3")

            try:
                model = whisper.load_model("base")
                result = model.transcribe(chunk_file, language="en")
            except Exception as e:
                logger.error(f"❌ Failed to transcribe chunk {i}: {e}")
                chunk_bar.update(1)
                continue

            segments = result.get("segments", [])
            with tqdm(
                total=len(segments),
                desc=f"Segments [Chunk {i+1}]",
                position=2,
                leave=False
            ) as segment_bar:

                for segment in segments:
                    english = segment["text"].strip()
                    if english == last_english:
                        segment_bar.update(1)
                        continue
                    last_english = english

                    start = timedelta(seconds=offset + segment["start"])
                    end = timedelta(seconds=offset + segment["end"])

                    try:
                        content = (
                            translate_text(english, language, api_key, provider, ollama_model)
                            if translate else english
                        )
                    except Exception as e:
                        content = "[Translation error]"
                        logger.error(f"[Chunk {i}] Translation error: {e}")

                    subtitle = srt.Subtitle(index=srt_index, start=start, end=end, content=content)
                    srt_file.write(srt.compose([subtitle]))
                    srt_index += 1
                    segment_bar.update(1)

            offset += (chunk_len - overlap) / 1000.0
            chunk_bar.update(1)

            try:
                os.remove(chunk_file)
            except Exception:
                logger.warning(f"⚠️ Could not remove temp chunk file: {chunk_file}")

    logger.info(f"✅ Subtitles saved to: {srt_path}")

def process_video_directory(directory_path, config):
    video_files = [
        f for f in os.listdir(directory_path)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
    ]

    with tqdm(total=len(video_files), desc="Videos", position=0) as video_bar:
        for filename in video_files:
            video_path = os.path.join(directory_path, filename)
            srt_path = os.path.splitext(video_path)[0] + ".srt"
            logger.info(f"Processing: {video_path}")
            process_single_video(video_path, srt_path, config, parent_bar=video_bar)
            video_bar.update(1)

def verify_or_retranslate_ollama(original_english, translated_text, target_language, model="llama3", max_retries=10):
    """Verifies if translation is in the correct script and retries if not."""
    verify_prompt = (
        f"You are a linguistic verification assistant.\n"
        f"Verify if the following sentence is entirely in {target_language} script and contains no English, romanization, or foreign words.\n"
        f"Respond with YES if it is valid {target_language}. Respond with NO if it contains any incorrect elements.\n\n"
        f"---\nSentence:\n{translated_text}\n\nAnswer:"
    )
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": verify_prompt, "stream": False}
            )
            response.raise_for_status()
            answer = response.json()["response"].strip().lower()
            logger.info(f"verified {translated_text}")
            if "yes" in answer:
                return translated_text
            else:
                logger.warning(f"[Verify Attempt {attempt + 1}] Invalid result. Retrying translation...")
                translated_text = translate_with_ollama(original_english, target_language, model=model)
        except Exception as e:
            logger.error(f"Verification error: {e}")
            break
    return "[Translation verification failed]"

def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from video(s).")
    parser.add_argument("input", help="Path to a video file or directory")
    parser.add_argument("srt_output", help="Path to save .srt output or directory")
    parser.add_argument("--config", help="Path to config JSON", required=False)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}

    if os.path.isdir(args.input):
        if not os.path.exists(args.srt_output):
            os.makedirs(args.srt_output)
        for file in os.listdir(args.input):
            if file.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
                input_path = os.path.join(args.input, file)
                output_path = os.path.join(args.srt_output, os.path.splitext(file)[0] + ".srt")
                process_single_video(input_path, output_path, config)
    else:
        process_single_video(args.input, args.srt_output, config)

if __name__ == "__main__":
    logger.info("Starting subtitle generation...")
    main()
