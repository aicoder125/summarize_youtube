# YouTube Video Summarizer

A Python tool that downloads YouTube video subtitles and generates summaries using either local Ollama or OpenAI API.

## Features

- Downloads subtitles from YouTube videos using `yt-dlp`
- Converts VTT subtitle files to clean text
- Summarizes content using:
  - **Ollama** (local, free) - default option
  - **OpenAI API** (cloud-based, requires API key)

## Prerequisites

- Python 3.8 or higher
- `yt-dlp` installed on your system
- For Ollama: Ollama installed and running locally
- For OpenAI: An OpenAI API key

## Setup

### 1. Clone or navigate to the project directory

```bash
cd summarize_youtube
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install yt-dlp (if not already installed)

**On macOS:**
```bash
brew install yt-dlp
```

**On Linux:**
```bash
sudo apt-get install yt-dlp
# or
pip install yt-dlp
```

**On Windows:**
```bash
pip install yt-dlp
```

### 6. (Optional) Set up OpenAI API key

If you want to use OpenAI instead of Ollama, create a `.env` file in the project directory:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Or export it as an environment variable:
```bash
export OPENAI_API_KEY=your-api-key-here
```

### 7. (Optional) Set up Ollama

If you want to use Ollama (default):

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the required model:
   ```bash
   ollama pull qwen2.5:7b
   ```

## Usage

### Basic usage (with Ollama - default)

```bash
python summarize_youtube.py <video_id_or_url>
```

Examples:
```bash
# Using video ID
python summarize_youtube.py dQw4w9WgXcQ

# Using full YouTube URL
python summarize_youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

### Using OpenAI API

```bash
python summarize_youtube.py <video_id_or_url> --openai
```

### Using a specific model

```bash
# With Ollama
python summarize_youtube.py <video_id_or_url> --ollama --model llama2:7b

# With OpenAI
python summarize_youtube.py <video_id_or_url> --openai --model gpt-4o-mini
```

### Getting the YouTube video ID

You can use either:
- **Full URL**: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- **Video ID only**: `dQw4w9WgXcQ` (the part after `v=` in the URL)

The script automatically handles both formats.

## Output

The script will:
1. Download subtitles and save them as `{video_id}.en.vtt`
2. Convert subtitles to text and save as `{video_id}.en.txt`
3. Generate a summary and save it as `{video_id}_summary.txt`
4. Display the summary in the terminal

## Troubleshooting

### "yt-dlp: command not found"
- Make sure `yt-dlp` is installed and in your PATH
- Try installing it with pip: `pip install yt-dlp`

### "Failed to download subtitles" or SABR streaming warnings
- The script automatically tries multiple strategies to work around YouTube's SABR streaming issues
- If it still fails, try updating yt-dlp to the latest version:
  ```bash
  pip install -U yt-dlp
  ```
- Make sure the video has subtitles available (check in your browser)
- Some videos may not have subtitles enabled

### "Ollama error: Connection refused"
- Make sure Ollama is running: `ollama serve`
- Check if the model is installed: `ollama list`

### "OPENAI_API_KEY not found"
- Create a `.env` file with your API key, or
- Export it as an environment variable

### "No subtitles found"
- The video may not have subtitles available
- Try a different video or check if auto-generated subtitles are enabled
- The script will try multiple methods automatically, but some videos simply don't have subtitles

## Requirements

See `requirements.txt` for the full list of Python dependencies:
- `yt-dlp` - YouTube subtitle downloader
- `ollama` - Local LLM client
- `python-dotenv` - Environment variable management
- `openai` - OpenAI API client

## License

This project is provided as-is for educational and personal use.
