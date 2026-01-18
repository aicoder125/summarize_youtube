#!/usr/bin/env python3
"""
YouTube Video Summarizer - Web Interface

A Flask web app that allows users to paste a YouTube URL and get a summary.
"""

import os
import re
import json
import subprocess
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify
import ollama
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# HTML template for the main page
MAIN_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Video Summarizer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 48px;
            max-width: 600px;
            width: 100%;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 8px;
            color: #333;
        }
        .subtitle {
            color: #666;
            margin-bottom: 32px;
        }
        .input-group {
            margin-bottom: 24px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #444;
        }
        input[type="text"] {
            width: 100%;
            padding: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .model-select {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        .model-option {
            flex: 1;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .model-option:hover {
            border-color: #667eea;
        }
        .model-option.selected {
            border-color: #667eea;
            background: #f0f3ff;
        }
        .model-option input {
            display: none;
        }
        button {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 24px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e0e0e0;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .loading-text {
            color: #666;
        }
        .error {
            background: #fee;
            color: #c00;
            padding: 16px;
            border-radius: 12px;
            margin-top: 24px;
            display: none;
        }
        .error.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>YouTube Summarizer</h1>
        <p class="subtitle">Paste a YouTube URL and get an AI-powered summary</p>

        <form id="summarize-form">
            <div class="input-group">
                <label for="url">YouTube URL or Video ID</label>
                <input type="text" id="url" name="url"
                    placeholder="https://www.youtube.com/watch?v=... or video ID"
                    required>
            </div>

            <label>Subtitle Language</label>
            <div class="model-select">
                <label class="model-option selected">
                    <input type="radio" name="language" value="english" checked>
                    <div>English</div>
                </label>
                <label class="model-option">
                    <input type="radio" name="language" value="chinese">
                    <div>Chinese (zh-Hans)</div>
                </label>
            </div>

            <label>Summarization Model</label>
            <div class="model-select">
                <label class="model-option selected">
                    <input type="radio" name="model" value="ollama" checked>
                    <div>Ollama (Local)</div>
                </label>
                <label class="model-option">
                    <input type="radio" name="model" value="openai">
                    <div>OpenAI API</div>
                </label>
            </div>

            <div class="checkbox-group" style="margin-bottom: 24px;">
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" name="transcribe" id="transcribe" style="width: 18px; height: 18px;">
                    <span>Transcribe audio if no subtitles (uses Whisper, slower)</span>
                </label>
            </div>

            <button type="submit" id="submit-btn">Summarize</button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p class="loading-text" id="loading-text">Downloading subtitles...</p>
        </div>

        <div class="error" id="error"></div>
    </div>

    <script>
        // Model selection UI
        document.querySelectorAll('.model-option').forEach(option => {
            option.addEventListener('click', () => {
                document.querySelectorAll('.model-option').forEach(o => o.classList.remove('selected'));
                option.classList.add('selected');
            });
        });

        // Form submission
        document.getElementById('summarize-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const url = document.getElementById('url').value;
            const model = document.querySelector('input[name="model"]:checked').value;
            const language = document.querySelector('input[name="language"]:checked').value;
            const transcribe = document.getElementById('transcribe').checked;
            const submitBtn = document.getElementById('submit-btn');
            const loading = document.getElementById('loading');
            const loadingText = document.getElementById('loading-text');
            const error = document.getElementById('error');

            // Reset state
            error.classList.remove('show');
            loading.classList.add('show');
            submitBtn.disabled = true;

            // Update loading text
            const steps = transcribe ? [
                'Downloading subtitles...',
                'No subtitles found, downloading audio...',
                'Transcribing audio with Whisper...',
                'Generating summary with AI...',
                'Almost done...'
            ] : [
                'Downloading subtitles...',
                'Converting to text...',
                'Generating summary with AI...',
                'Almost done...'
            ];
            let stepIndex = 0;
            const stepInterval = setInterval(() => {
                stepIndex = Math.min(stepIndex + 1, steps.length - 1);
                loadingText.textContent = steps[stepIndex];
            }, 3000);

            try {
                const response = await fetch('/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, model, language, transcribe })
                });

                const data = await response.json();

                clearInterval(stepInterval);

                if (data.success) {
                    // Open result in new tab
                    const newTab = window.open('', '_blank');
                    newTab.document.write(data.html);
                    newTab.document.close();
                } else {
                    error.textContent = data.error || 'An error occurred';
                    error.classList.add('show');
                }
            } catch (err) {
                clearInterval(stepInterval);
                error.textContent = 'Failed to connect to server: ' + err.message;
                error.classList.add('show');
            } finally {
                loading.classList.remove('show');
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''

# HTML template for the result page
RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary: {title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .video-header {{
            position: relative;
            background: #000;
        }}
        .thumbnail {{
            width: 100%;
            height: auto;
            display: block;
            opacity: 0.9;
        }}
        .video-info {{
            padding: 24px;
            background: linear-gradient(to bottom, #1a1a2e, #16213e);
            color: white;
        }}
        .video-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .video-meta {{
            display: flex;
            gap: 16px;
            color: #aaa;
            font-size: 0.9rem;
            flex-wrap: wrap;
        }}
        .video-meta a {{
            color: #667eea;
            text-decoration: none;
        }}
        .video-meta a:hover {{
            text-decoration: underline;
        }}
        .summary-section {{
            padding: 32px;
        }}
        .summary-header {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #eee;
        }}
        .summary-content p {{
            margin-bottom: 12px;
            text-align: justify;
        }}
        .summary-content strong {{
            color: #444;
        }}
        .summary-content .list-item {{
            padding-left: 8px;
        }}
        .summary-content .bullet {{
            padding-left: 20px;
        }}
        .footer {{
            padding: 20px 32px;
            background: #f8f9fa;
            text-align: center;
            color: #666;
            font-size: 0.85rem;
        }}
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="video-header">
            <img class="thumbnail" src="{thumbnail}" alt="Video thumbnail">
        </div>
        <div class="video-info">
            <h1 class="video-title">{title}</h1>
            <div class="video-meta">
                <span>{channel}</span>
                <span>{duration}</span>
                <a href="{url}" target="_blank">Watch on YouTube</a>
            </div>
        </div>
        <div class="summary-section">
            <h2 class="summary-header">Summary</h2>
            <div class="summary-content">
                {summary_html}
            </div>
        </div>
        <div class="footer">
            Generated by YouTube Summarizer
        </div>
    </div>
</body>
</html>
'''


def extract_video_id(video_input: str) -> str:
    """Extract video ID from YouTube URL or return as-is if already an ID."""
    if not video_input.startswith(('http://', 'https://')):
        return video_input

    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*[&?]v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, video_input)
        if match:
            return match.group(1)
    return video_input


def download_subtitles(video_id: str, use_chinese: bool = False) -> str | None:
    """Download subtitles using yt-dlp."""
    if use_chinese:
        # Chinese language strategies
        strategies = [
            ["yt-dlp", "--impersonate", "chrome", "--write-subs", "--write-auto-subs", "--sub-lang", "zh-Hans", "--skip-download", "--ignore-errors", video_id, "-o", f"{video_id}.%(ext)s"],
            ["yt-dlp", "--impersonate", "chrome", "--write-subs", "--write-auto-subs", "--sub-lang", "zh-CN", "--skip-download", "--ignore-errors", video_id, "-o", f"{video_id}.%(ext)s"],
            ["yt-dlp", "--impersonate", "chrome", "--write-subs", "--write-auto-subs", "--sub-lang", "zh", "--skip-download", "--ignore-errors", video_id, "-o", f"{video_id}.%(ext)s"],
            ["yt-dlp", "--impersonate", "chrome", "--write-auto-subs", "--sub-lang", "zh-Hans,zh-CN,zh", "--skip-download", "--ignore-errors", video_id, "-o", f"{video_id}.%(ext)s"],
        ]
        patterns = [f"{video_id}.zh-Hans.vtt", f"{video_id}.zh-CN.vtt", f"{video_id}.zh.vtt", f"{video_id}.zh-*.vtt", f"{video_id}.*.vtt"]
    else:
        # English language strategies
        strategies = [
            ["yt-dlp", "--impersonate", "chrome", "--write-auto-subs", "--skip-download", "--ignore-errors", video_id, "-o", f"{video_id}.%(ext)s"],
            ["yt-dlp", "--impersonate", "chrome", "--write-subs", "--write-auto-subs", "--sub-lang", "en", "--skip-download", "--ignore-errors", video_id, "-o", f"{video_id}.%(ext)s"],
            ["yt-dlp", "--impersonate", "chrome", "--write-subs", "--sub-lang", "en-US", "--skip-download", "--ignore-errors", video_id, "-o", f"{video_id}.%(ext)s"],
        ]
        patterns = [f"{video_id}.en.vtt", f"{video_id}.en-US.vtt", f"{video_id}.en-en-US.vtt", f"{video_id}.*.vtt"]

    for cmd in strategies:
        subprocess.run(cmd, capture_output=True, text=True)

        for pattern in patterns:
            matches = list(Path(".").glob(pattern))
            if matches:
                return str(matches[0])
    return None


def clean_vtt_to_text(vtt_path: str) -> str:
    """Convert VTT file to clean text."""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    clean_text = []
    last_line = ""

    for line in lines:
        line = re.sub(r'\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}.*?\n', '', line)
        line = re.sub(r'<[^>]+>', '', line)
        line = line.replace('WEBVTT', '').replace('Kind: captions', '')
        line = line.replace('Language: en', '').replace('Language: zh-Hans', '')
        line = line.replace('Language: zh-CN', '').replace('Language: zh', '')
        line = line.strip()
        if line and line != last_line:
            clean_text.append(line)
            last_line = line

    return '\n'.join(clean_text)


def download_audio(video_id: str) -> str | None:
    """Download audio from YouTube video using yt-dlp."""
    audio_file = f"{video_id}.mp3"

    cmd = [
        "yt-dlp",
        "--impersonate", "chrome",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", audio_file,
        "--no-playlist",
        video_id
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(audio_file):
        return audio_file

    for ext in ['.m4a', '.webm', '.opus', '.wav']:
        alt_file = f"{video_id}{ext}"
        if os.path.exists(alt_file):
            return alt_file

    return None


def transcribe_audio(audio_path: str, language: str = "en") -> str | None:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper

        model_size = "medium" if language == "zh" else "base"
        model = whisper.load_model(model_size)

        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False
        )

        return result["text"].strip()

    except ImportError:
        return None
    except Exception:
        return None


def get_video_metadata(video_id: str) -> dict:
    """Get video metadata using yt-dlp."""
    cmd = ["yt-dlp", "--impersonate", "chrome", "--dump-json", "--skip-download", video_id]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            return {
                "title": data.get("title", "Unknown Title"),
                "channel": data.get("channel", data.get("uploader", "Unknown Channel")),
                "thumbnail": data.get("thumbnail", ""),
                "duration": data.get("duration_string", ""),
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
        except json.JSONDecodeError:
            pass

    return {
        "title": "Unknown Title",
        "channel": "Unknown Channel",
        "thumbnail": "",
        "duration": "",
        "url": f"https://www.youtube.com/watch?v={video_id}"
    }


def summarize_with_ollama(text: str, model: str = "qwen2.5:7b") -> str:
    """Summarize text using local Ollama."""
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Please summarize the main points:\n\n{text}"
        }]
    )
    return response['message']['content']


def summarize_with_openai(text: str, model: str = "gpt-4o-mini") -> str:
    """Summarize text using OpenAI API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Please summarize the main points:\n\n{text}"
        }]
    )
    return response.choices[0].message.content


def translate_to_chinese(text: str, use_openai: bool = False, model: str = None) -> str:
    """
    Translate text to Simplified Chinese using LLM.
    Uses Qwen 2.5 (Ollama) by default - good for Chinese translation.
    """
    prompt = f"""Translate the following text to Simplified Chinese (简体中文).
Output ONLY the Chinese translation, no explanations or original text.

Text to translate:
{text}"""

    if use_openai:
        from openai import OpenAI

        model = model or "gpt-4o-mini"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    else:
        model = model or "qwen2.5:7b"
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']


def format_summary_html(summary: str) -> str:
    """Convert summary text to HTML."""
    summary_html = ""
    for line in summary.split("\n"):
        line = line.strip()
        if line:
            # Convert **bold** to <strong>bold</strong>
            line = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', line)

            if re.match(r'^\d+\.', line):
                summary_html += f"<p class='list-item'>{line}</p>\n"
            elif line.startswith(('- ', '* ', '• ')):
                summary_html += f"<p class='bullet'>{line}</p>\n"
            else:
                summary_html += f"<p>{line}</p>\n"
    return summary_html


@app.route('/')
def index():
    """Main page."""
    return render_template_string(MAIN_PAGE)


@app.route('/summarize', methods=['POST'])
def summarize():
    """Process video and return summary."""
    try:
        data = request.json
        url = data.get('url', '').strip()
        model_type = data.get('model', 'ollama')
        language = data.get('language', 'english')
        use_chinese = language == 'chinese'
        use_transcribe = data.get('transcribe', False)
        use_openai = model_type == 'openai'

        if not url:
            return jsonify({'success': False, 'error': 'Please enter a YouTube URL'})

        video_id = extract_video_id(url)

        # Track if we need to translate to Chinese
        text = None
        needs_translation = False

        if use_chinese:
            # Step 1: Try Chinese subtitles first
            vtt_file = download_subtitles(video_id, use_chinese=True)

            if vtt_file:
                text = clean_vtt_to_text(vtt_file)

            # Step 2: Fallback to English subtitles + translation
            if not text:
                vtt_file = download_subtitles(video_id, use_chinese=False)

                if vtt_file:
                    text = clean_vtt_to_text(vtt_file)
                    needs_translation = True

            # Step 3: Fallback to audio transcription
            if not text and use_transcribe:
                audio_file = download_audio(video_id)
                if audio_file:
                    # Try Chinese transcription first
                    text = transcribe_audio(audio_file, language="zh")
                    if not text:
                        # If Chinese transcription fails, try English and translate
                        text = transcribe_audio(audio_file, language="en")
                        if text:
                            needs_translation = True
        else:
            # Original English flow
            vtt_file = download_subtitles(video_id, use_chinese=False)

            if vtt_file:
                text = clean_vtt_to_text(vtt_file)

            # If no subtitles, try transcription if enabled
            if not text and use_transcribe:
                audio_file = download_audio(video_id)
                if audio_file:
                    text = transcribe_audio(audio_file, language="en")

        if not text:
            error_msg = 'Failed to download subtitles. The video may not have captions available.'
            if not use_transcribe:
                error_msg += ' Try enabling "Transcribe audio" option.'
            return jsonify({'success': False, 'error': error_msg})

        # Translate to Chinese if needed
        if needs_translation and text:
            text = translate_to_chinese(text, use_openai=use_openai)

        # Get metadata
        metadata = get_video_metadata(video_id)

        # Summarize
        if model_type == 'openai':
            summary = summarize_with_openai(text)
        else:
            summary = summarize_with_ollama(text)

        # Generate HTML
        summary_html = format_summary_html(summary)
        result_html = RESULT_TEMPLATE.format(
            title=metadata['title'],
            channel=metadata['channel'],
            thumbnail=metadata['thumbnail'],
            duration=metadata['duration'],
            url=metadata['url'],
            summary_html=summary_html
        )

        return jsonify({'success': True, 'html': result_html})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("=" * 50)
    print("YouTube Video Summarizer")
    print("=" * 50)
    print()
    print("Open http://localhost:5001 in your browser")
    print("Press Ctrl+C to stop the server")
    print()
    app.run(debug=True, port=5001)
