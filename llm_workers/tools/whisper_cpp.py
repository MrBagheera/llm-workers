import logging
import os
from typing import Annotated, List

from langchain_core.tools import InjectedToolArg, StructuredTool, ToolException

from llm_workers.tools.transcribe_common import Segment, merge_segments
from llm_workers.tools.utils import run_process, cached

logger = logging.getLogger(__name__)

import re

_whisper_segment_pattern = re.compile(
    r'^\[(\d+):(\d+):(\d+\.\d+)\s-->\s(\d+):(\d+):(\d+\.\d+)\]\s\s(.*)$'
)

def _convert_whisper_output_to_segment(line):
    match = _whisper_segment_pattern.match(line)
    if match:
        h1, m1, s1, h2, m2, s2, text = match.groups()
        if len(text) == 0:
            return None
        start = float(h1)*3600 + float(m1)*60 + float(s1)
        end = float(h2)*3600 + float(m2)*60 + float(s2)
        return Segment(start, end, text)
    return None


def _make_transcript(
    file: str,
    model: Annotated[str, InjectedToolArg],
    extra_args: Annotated[str, InjectedToolArg] = None,
    max_delay: Annotated[float, InjectedToolArg] = 1,
    max_length: Annotated[int, InjectedToolArg] = 150,
) -> str:
    """
    Transcribe an audio file to text using ffmpeg and whisper.cpp (must be pre-installed).

    Args:
        file: file to read audio from. If not WAV file, it'll be converted to WAV using ffmpeg.
        model: model to use for whisper.cpp
        extra_args: additional arguments to pass to whisper-cli
        max_delay: maximum delay between segments to merge them
        max_length: maximum length of segment
    """

    # convert to WAV
    wav_path = cached(
        file,
        ".wav",
        lambda cache_path: run_process([
            "ffmpeg",
            "-i", file,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            cache_path
        ])
    )

    # TODO run pyannote if needed

    # run whisper
    def _run_whisper(cache_path: str):
        whisper_args = [
            "whisper-cli",
            "--model", model,
            "--file", wav_path,
            "--max-len", "1",
        ]
        if extra_args is not None:
            whisper_args.extend(extra_args.split())
        segments = run_process(whisper_args)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(segments)
    raw_transcript_path = cached(
        wav_path,
        ".raw.txt",
        _run_whisper,
        discriminator=f"{model}/{extra_args}")

    # read and merge segments
    def _read_and_merge_segments(cache_path: str):
        segments = []
        with open(raw_transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                segment = _convert_whisper_output_to_segment(line)
                if segment is not None:
                    segments.append(segment)
        segments = merge_segments(segments, max_delay, max_length)
        with open(cache_path, "w", encoding="utf-8") as f:
            for segment in segments:
                f.write(segment.format())
    transcript_path = cached(
        raw_transcript_path,
        ".final.txt",
        _read_and_merge_segments,
        discriminator=f"{max_delay}/{max_length}"
    )

    # read final transcript
    with open(transcript_path, 'r') as file:
        transcript = file.read()
    return transcript


make_transcript = StructuredTool.from_function(
    _make_transcript,
    name="whisper_cpp",
    parse_docstring=True,
    error_on_invalid_docstring=True
)
