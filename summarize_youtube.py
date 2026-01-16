#!/usr/bin/env python3
"""
YouTube Video Summarizer

Downloads YouTube subtitles, converts to text, and summarizes using LLM.
Supports both local Ollama and OpenAI API.

Usage:
    python summarize_youtube.py <video_id>              # Use Ollama (default)
    python summarize_youtube.py <video_id> --openai    # Use OpenAI API
    python summarize_youtube.py <video_id> --ollama    # Use Ollama explicitly
    python summarize_youtube.py <video_id> --html      # Generate HTML and open in browser
"""

import argparse
import json
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

import ollama
from dotenv import load_dotenv

load_dotenv()


def extract_video_id(video_input: str) -> str:
    """
    Extract video ID from YouTube URL or return as-is if already an ID.

    Args:
        video_input: YouTube URL or video ID

    Returns:
        Video ID
    """
    # If it's already a video ID (no http/https), return as-is
    if not video_input.startswith(('http://', 'https://')):
        return video_input

    # Extract video ID from various YouTube URL formats
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*[&?]v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, video_input)
        if match:
            return match.group(1)
    
    # If no pattern matches, try to use the input as-is
    return video_input


def check_subtitles_available(video_id: str) -> bool:
    """
    Check if subtitles are available for the video.

    Args:
        video_id: YouTube video ID

    Returns:
        True if subtitles are available, False otherwise
    """
    cmd = [
        "yt-dlp",
        "--impersonate", "chrome",
        "--list-subs",
        "--skip-download",
        video_id
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if any subtitles are listed
    output = result.stdout.lower()
    return "language" in output or "available" in output or "en" in output


def download_subtitles(video_input: str) -> str | None:
    """
    Download subtitles using yt-dlp with multiple fallback strategies.

    Args:
        video_input: YouTube video ID or URL

    Returns:
        Path to VTT file, or None if download fails
    """
    video_id = extract_video_id(video_input)
    print(f"Downloading subtitles for: {video_id}...")

    # First, check if subtitles are available
    print("  Checking if subtitles are available...")
    if not check_subtitles_available(video_id):
        print("  ⚠ Warning: No subtitles detected. Trying anyway...")
    else:
        print("  ✓ Subtitles detected")

    # Strategy 1: Standard approach - let yt-dlp auto-detect best subtitle language
    strategies = [
        {
            "name": "Auto-detect English subtitles (no language specified)",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-auto-subs",
                "--skip-download",
                "--ignore-errors",
                video_id,
                "-o", f"{video_id}.%(ext)s"
            ]
        },
        {
            "name": "Standard (with SABR workaround)",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-subs",
                "--write-auto-subs",
                "--sub-lang", "en",
                "--skip-download",
                "--ignore-errors",
                video_id,
                "-o", f"{video_id}.%(ext)s"
            ]
        },
        {
            "name": "Try en-US (English United States)",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-subs",
                "--sub-lang", "en-US",
                "--skip-download",
                "--ignore-errors",
                video_id,
                "-o", f"{video_id}.%(ext)s"
            ]
        },
        {
            "name": "Try en-en-US (English from English)",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-auto-subs",
                "--sub-lang", "en-en-US",
                "--skip-download",
                "--ignore-errors",
                video_id,
                "-o", f"{video_id}.%(ext)s"
            ]
        },
        {
            "name": "With extractor args (web client)",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-auto-subs",
                "--skip-download",
                "--extractor-args", "youtube:player-client=web",
                "--ignore-errors",
                video_id,
                "-o", f"{video_id}.%(ext)s"
            ]
        },
        {
            "name": "With extractor args (tv client)",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-auto-subs",
                "--skip-download",
                "--extractor-args", "youtube:player-client=tv",
                "--ignore-errors",
                video_id,
                "-o", f"{video_id}.%(ext)s"
            ]
        },
        {
            "name": "With Android client (more compatible)",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-auto-subs",
                "--skip-download",
                "--extractor-args", "youtube:player-client=android",
                "--ignore-errors",
                video_id,
                "-o", f"{video_id}.%(ext)s"
            ]
        },
        {
            "name": "Full URL format",
            "cmd": [
                "yt-dlp",
                "--impersonate", "chrome",
                "--write-auto-subs",
                "--skip-download",
                "--ignore-errors",
                f"https://www.youtube.com/watch?v={video_id}",
                "-o", f"{video_id}.%(ext)s"
            ]
        }
    ]

    last_result = None
    for strategy in strategies:
        print(f"  Trying: {strategy['name']}...")
        result = subprocess.run(strategy['cmd'], capture_output=True, text=True)
        last_result = result

        # Check for VTT file with various patterns (including en-US, en-en-US and other variants)
        patterns = [
            f"{video_id}.en.vtt",
            f"{video_id}.en-US.vtt",
            f"{video_id}.en-en-US.vtt",
            f"{video_id}.*.vtt",
            f"{video_id}.vtt"
        ]
        
        for pattern in patterns:
            matches = list(Path(".").glob(pattern))
            if matches:
                vtt_file = str(matches[0])
                print(f"✓ Subtitles downloaded: {vtt_file}")
                return vtt_file

    # If all strategies failed, show error
    print(f"✗ Failed to download subtitles after trying {len(strategies)} strategies")
    print("  Possible reasons:")
    print("  - Video may not have subtitles available")
    print("  - Update yt-dlp: pip install -U yt-dlp")
    print("  - Check if video exists and is accessible")
    print("  - Try checking subtitles manually: yt-dlp --list-subs <video_id>")
    
    # Show more detailed error information
    if last_result:
        if last_result.stdout:
            stdout_lines = last_result.stdout.strip().split('\n')
            # Look for useful information in stdout
            useful_info = [line for line in stdout_lines if any(keyword in line.lower() for keyword in ['subtitle', 'caption', 'language', 'available'])]
            if useful_info:
                print("\n  yt-dlp output:")
                for line in useful_info[:5]:  # Show first 5 relevant lines
                    print(f"    {line}")
        
        if last_result.stderr:
            stderr_lines = last_result.stderr.strip().split('\n')
            # Filter out SABR warnings but show actual errors
            actual_errors = [line for line in stderr_lines if 'ERROR' in line.upper() or ('error' in line.lower() and 'warning' not in line.lower())]
            if actual_errors:
                print("\n  Errors:")
                for line in actual_errors[:3]:  # Show first 3 errors
                    print(f"    {line}")
    
    return None


def clean_vtt_to_text(vtt_path: str) -> str:
    """
    Convert VTT file to clean text.

    Args:
        vtt_path: Path to VTT file

    Returns:
        Clean text content
    """
    print(f"Converting VTT to text...")

    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    clean_text = []
    last_line = ""

    for line in lines:
        # Remove timestamps (e.g., 00:03:19.920 --> 00:03:22.880)
        line = re.sub(r'\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}.*?\n', '', line)
        # Remove HTML tags (e.g., <c>, </c>)
        line = re.sub(r'<[^>]+>', '', line)
        # Remove VTT headers
        line = line.replace('WEBVTT', '').replace('Kind: captions', '').replace('Language: en', '')
        # Strip whitespace
        line = line.strip()
        # Deduplicate (YouTube auto-subs repeat lines)
        if line and line != last_line:
            clean_text.append(line)
            last_line = line

    # Save to text file
    txt_path = vtt_path.replace('.vtt', '.txt')
    content = '\n'.join(clean_text)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Text saved: {txt_path} ({len(content)} characters)")
    return content


def summarize_with_ollama(text: str, model: str = "qwen2.5:7b") -> str:
    """
    Summarize text using local Ollama.

    Args:
        text: Text to summarize
        model: Ollama model name

    Returns:
        Summary text
    """
    print(f"Summarizing with Ollama ({model})...")

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Please summarize the main points:\n\n{text}"
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print(f"  And model is installed: ollama pull {model}")
        sys.exit(1)


def summarize_with_openai(text: str, model: str = "gpt-4o-mini") -> str:
    """
    Summarize text using OpenAI API.

    Args:
        text: Text to summarize
        model: OpenAI model name

    Returns:
        Summary text
    """
    print(f"Summarizing with OpenAI ({model})...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("✗ OPENAI_API_KEY not found in environment")
        print("  Set it in .env file or export OPENAI_API_KEY=your-key")
        sys.exit(1)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Please summarize the main points:\n\n{text}"
            }]
        )
        return response.choices[0].message.content
    except ImportError:
        print("✗ openai package not installed")
        print("  Run: pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"✗ OpenAI error: {e}")
        sys.exit(1)


def get_video_metadata(video_id: str) -> dict:
    """
    Get video metadata using yt-dlp.

    Args:
        video_id: YouTube video ID

    Returns:
        Dictionary with title, channel, thumbnail, duration
    """
    cmd = [
        "yt-dlp",
        "--impersonate", "chrome",
        "--dump-json",
        "--skip-download",
        video_id
    ]

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


def generate_html(video_id: str, summary: str, metadata: dict) -> str:
    """
    Generate HTML file with video summary.

    Args:
        video_id: YouTube video ID
        summary: Summary text
        metadata: Video metadata

    Returns:
        Path to generated HTML file
    """
    # Convert summary to HTML paragraphs
    summary_html = ""
    for line in summary.split("\n"):
        line = line.strip()
        if line:
            # Convert **bold** to <strong>bold</strong>
            line = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', line)

            # Handle numbered lists
            if re.match(r'^\d+\.', line):
                summary_html += f"<p class='list-item'>{line}</p>\n"
            # Handle bullet points
            elif line.startswith(('- ', '* ', '• ')):
                summary_html += f"<p class='bullet'>{line}</p>\n"
            else:
                summary_html += f"<p>{line}</p>\n"

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary: {metadata["title"]}</title>
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
            opacity: 0.8;
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
        .summary-content h3 {{
            margin: 20px 0 12px 0;
            color: #444;
            font-size: 1.1rem;
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
            <img class="thumbnail" src="{metadata["thumbnail"]}" alt="Video thumbnail">
        </div>
        <div class="video-info">
            <h1 class="video-title">{metadata["title"]}</h1>
            <div class="video-meta">
                <span>{metadata["channel"]}</span>
                <span>{metadata["duration"]}</span>
                <a href="{metadata["url"]}" target="_blank">Watch on YouTube</a>
            </div>
        </div>
        <div class="summary-section">
            <h2 class="summary-header">Summary</h2>
            <div class="summary-content">
                {summary_html}
            </div>
        </div>
        <div class="footer">
            Generated by <a href="https://github.com/anthropics/claude-code">YouTube Summarizer</a>
        </div>
    </div>
</body>
</html>'''

    html_file = f"{video_id}_summary.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_file


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube subtitles and generate summary"
    )
    parser.add_argument("video_id", help="YouTube video ID or full URL (e.g., dQw4w9WgXcQ or https://www.youtube.com/watch?v=dQw4w9WgXcQ)")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API for summarization")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for summarization (default)")
    parser.add_argument("--model", help="Model to use (default: gpt-4o-mini for OpenAI, qwen2.5:7b for Ollama)")
    parser.add_argument("--html", action="store_true", help="Generate HTML output and open in browser")

    args = parser.parse_args()

    print("=" * 60)
    print("YouTube Video Summarizer")
    print("=" * 60)
    print()

    # Extract video ID from URL if needed
    video_id = extract_video_id(args.video_id)

    # Step 1: Download subtitles
    vtt_file = download_subtitles(video_id)
    if not vtt_file:
        sys.exit(1)

    print()

    # Step 2: Convert VTT to text
    text = clean_vtt_to_text(vtt_file)
    if not text:
        print("✗ No text content extracted")
        sys.exit(1)

    print()

    # Step 3: Summarize
    if args.openai:
        model = args.model or "gpt-4o-mini"
        summary = summarize_with_openai(text, model)
    else:
        model = args.model or "qwen2.5:7b"
        summary = summarize_with_ollama(text, model)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)
    print()

    # Save summary
    summary_file = f"{video_id}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✓ Summary saved: {summary_file}")

    # Generate HTML and open in browser if requested
    if args.html:
        print()
        print("Generating HTML output...")
        metadata = get_video_metadata(video_id)
        html_file = generate_html(video_id, summary, metadata)
        print(f"✓ HTML saved: {html_file}")
        print("Opening in browser...")
        webbrowser.open(f"file://{os.path.abspath(html_file)}")


if __name__ == "__main__":
    main()
