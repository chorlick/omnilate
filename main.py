import sys
import whisper
import requests
import srt
import datetime
import json 
import openai
import jsonschema
import logging
import os
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

def split_audio(input_path, chunk_ms=60000):
    audio = AudioSegment.from_file(input_path)
    chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]
    return chunks

def translate_text(text, target_language, api_key=None, provider="openai", ollama_model="llama3"):
    if provider == "openai":
        return translate_with_openai(text, target_language, api_key)
    elif provider == "ollama":
        return translate_with_ollama(text, target_language, model=ollama_model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def translate_with_openai(text, target_language, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": (
                "You are a professional translation engine. "
                "Translate the following English sentence into **{target_language}** only. Do not use any other language except the one provide. "
                "You must translate from english to **{target_language}**. Do not confuse one language with another. Check and verify your work after you are done"
                "Respond with ONLY the Thai sentence, no extra commentary."
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
        f"You are a translation assistant. Your task is to return ONLY the translated sentence "
        f"from English to {target_language}. Do NOT explain, annotate, or add any other content. "
        f"Input:\n{text}"
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        result = response.json()["response"].strip().strip('"').strip("“”")
        return result
    except Exception as e:
        raise RuntimeError(f"Ollama translation error: {str(e)}")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load and validate against schema
    schema_path = os.path.join(os.path.dirname(__file__), "schemas/pysub.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as ve:
        print(f"❌ Config validation error:\n{ve.message}")
        exit(1)

    return config



def extract_audio(video_path, audio_path="temp_audio.mp3"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

def transcribe(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    return result

def write_srt(transcription_result, output_path, translate=False, language="thai", api_key=None, provider="openai", ollama_model="llama3"):
    subtitles = []

    for i, segment in enumerate(transcription_result["segments"]):
        start = datetime.timedelta(seconds=segment["start"])
        end = datetime.timedelta(seconds=segment["end"])
        english = segment["text"].strip()

        if translate:
            try:
                translation = translate_text(english, language, api_key, provider, ollama_model)
                logger.info(f"[Line {i + 1}] [{provider}] EN: {english} → {language.upper()}: {translation}")
            except openai.OpenAIError as e:
                translation = "[Translation error]"
                logger.error(f"[Line {i + 1}] Failed to translate: '{english}'. Error: {e.__class__.__name__} - {str(e).splitlines()[0]}")
            except Exception as e:
                translation = "[Translation error]"
                logger.error(f"[Line {i + 1}] Unexpected error on text: '{english}'. Error: {e.__class__.__name__} - {str(e)}")

            content = translation
        else:
            content = english
            logger.info(f"[Line {i + 1}] EN only: {english}")

        subtitle = srt.Subtitle(index=i + 1, start=start, end=end, content=content)
        subtitles.append(subtitle)

    srt_data = srt.compose(subtitles)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_data)


def main():
    import argparse
    import os
    from pydub import AudioSegment
    import datetime
    import srt
    import whisper

    parser = argparse.ArgumentParser(description="Generate subtitles from a video.")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("srt_output_path", help="Path to output .srt file")
    parser.add_argument("--config", help="Path to JSON config file", required=False)

    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    translate = config.get("translate", False)
    target_language = config.get("target_language", "thai")
    api_key = config.get("api_key", None)
    provider = config.get("provider", "openai")
    ollama_model = config.get("ollama_model", "llama3")
    chunk_duration_sec = config.get("chunk_duration_sec", 60)
    chunk_overlap_sec = config.get("chunk_overlap_sec", 5)

    video_path = args.video_path
    srt_path = args.srt_output_path
    audio_path = "temp_audio.mp3"

    logger.info("Extracting audio...")
    extract_audio(video_path, audio_path)

    logger.info("Splitting audio into overlapping chunks...")
    chunk_len = chunk_duration_sec * 1000
    overlap = chunk_overlap_sec * 1000
    audio = AudioSegment.from_file(audio_path)

    chunks = []
    for i in range(0, len(audio), chunk_len - overlap):
        chunks.append(audio[i:i + chunk_len])

    srt_index = 1
    cumulative_offset = 0.0
    last_english = None

    logger.info(f"Processing {len(chunks)} chunks...")

    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, chunk in enumerate(chunks):
            chunk_file = f"chunk_{i}.mp3"
            chunk.export(chunk_file, format="mp3")
            logger.info(f"→ Chunk {i + 1}/{len(chunks)}: Transcribing...")

            try:
                model = whisper.load_model("base")
                result = model.transcribe(chunk_file, language="en")
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i + 1}: {e}")
                continue

            for segment in result["segments"]:
                english = segment["text"].strip()
                if english == last_english:
                    continue
                last_english = english

                start = datetime.timedelta(seconds=cumulative_offset + segment["start"])
                end = datetime.timedelta(seconds=cumulative_offset + segment["end"])

                if translate:
                    try:
                        translation = translate_text(
                            english, target_language, api_key, provider, ollama_model
                        )
                        logger.info(f"[Chunk {i + 1}] EN: {english} → {target_language.upper()}: {translation}")
                        content = translation
                    except Exception as e:
                        content = "[Translation error]"
                        logger.error(f"[Chunk {i + 1}] Error translating: {english} — {e}")
                else:
                    content = english
                    logger.info(f"[Chunk {i + 1}] EN: {english}")

                subtitle = srt.Subtitle(index=srt_index, start=start, end=end, content=content)
                srt_file.write(srt.compose([subtitle]))
                srt_index += 1

            cumulative_offset += (chunk_len - overlap) / 1000.0

            try:
                os.remove(chunk_file)
            except OSError:
                logger.warning(f"Could not delete temporary chunk file: {chunk_file}")

    logger.info(f"✅ Subtitles saved to: {srt_path}")

    
if __name__ == "__main__":
    main()