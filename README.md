# 🎬 PySub: Auto-Transcribe and Translate Subtitles from Video

**PySub** is a command-line utility that transcribes audio from video files and optionally translates the text into another language using either [OpenAI](https://platform.openai.com/) or [Ollama](https://ollama.com/) as the language model provider. Subtitles are exported in `.srt` format and are fully timestamped.

---

## ✨ Features

- 🎧 Audio extraction from `.mp4` videos
- 📝 Automatic English transcription using OpenAI Whisper
- 🌐 Optional translation into other languages (e.g. Thai, Isan, etc.)
- 🔄 Switch between OpenAI or Ollama (local LLM) via config file
- 📄 Outputs clean `.srt` subtitle files
- 📦 JSON-based configuration with schema validation
- 📋 Logging with translation line tracking and error handling

---

## 📂 Example Usage

```bash
python main.py input_video.mp4 output_subtitles.srt --config config.json
```

---

## 🛠️ Configuration

All settings are provided via a JSON config file. Here's an example `config.json`:

```json
{
  "translate": true,
  "target_language": "thai",
  "api_key": "sk-...",                 // Only needed for OpenAI
  "provider": "ollama",               // "openai" or "ollama"
  "ollama_model": "gemma:7b"          // Optional, defaults to "llama3"
}
```

---

## 📜 Example Output (.srt)

```srt
1
00:00:00,000 --> 00:00:03,000
ขอบคุณที่อยู่กับฉัน

2
00:00:03,001 --> 00:00:06,000
ฉันรักคุณมาก
```

---

## 📦 Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
jsonschema==4.24.0
moviepy==2.2.1
openai==1.86.0
openai_whisper==20240930
Requests==2.32.4
srt==3.5.3
```

---

## 🔧 Local Ollama Setup

If using `provider: "ollama"` in your config:

1. Install [Ollama](https://ollama.com/)
2. Pull a supported model (e.g. `gemma:7b` or `llama3`):
   ```bash
   ollama pull gemma:7b
   ollama run gemma:7b
   ```
3. Ensure it's running at `http://localhost:11434`

---

## 🛡️ API Key Safety

Never commit API keys to your repo. Use `.gitignore`, environment variables, or secured config files.  
If a key has leaked in Git history, refer to [Removing sensitive data from Git](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

---

## 📁 Folder Structure

```
├── main.py
├── config.json
├── requirements.txt
├── schemas/
│   └── pysub.schema.json
├── output.srt
├── pysub.log
```

---

## 🚀 Roadmap

- [ ] Batch processing of multiple video files
- [ ] GUI wrapper
- [ ] Translation memory / caching
- [ ] `.vtt` subtitle support

---

## 📄 License

MIT License © 2025 Christopher M. Horlick  
This project is not affiliated with OpenAI or Ollama.

---