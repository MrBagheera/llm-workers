import logging
from typing import Annotated, List

from langchain_core.tools import InjectedToolArg, StructuredTool, ToolException

from llm_workers.tools.transcribe import Segment, merge_segments, write_segments, read_segments, assign_speakers, \
    build_phrases
from llm_workers.tools.transcribe_pyannote import prepare_pyannote, diarize_and_write
from llm_workers.tools.utils import run_process, cached, multi_cached

logger = logging.getLogger(__name__)

import re

def _make_transcript(
    file: str,
    model: Annotated[str, InjectedToolArg],
    skip_seconds: int = None,
    length_seconds: int = None,
    detect_speakers: Annotated[bool, InjectedToolArg] = True,
    prompt: Annotated[str, InjectedToolArg] = None,
    extra_args: Annotated[str, InjectedToolArg] = None,
    max_delay: Annotated[float, InjectedToolArg] = 1,
    max_length: Annotated[int, InjectedToolArg] = 400,
    write_timestamps: Annotated[bool, InjectedToolArg] = False,
) -> str:
    """
    Transcribe an audio file to text using ffmpeg and whisper.cpp (must be pre-installed).

    Args:
        file: file to read audio from
        model: model to use for whisper.cpp
        skip_seconds: (optional) number of seconds to skip from the beginning of the audio file
        length_seconds: (optional) length of audio segment in seconds
        detect_speakers: (optional) if True, use pyannote to provide speaker information
        prompt: (optional) prompt to use for whisper, defaults to "Split text into individual sentences. Use capital letters and punctuation."
        extra_args: additional arguments to pass to whisper-cli
        max_delay: maximum delay between segments that can still be merged
        max_length: maximum length of merged segment
        write_timestamps: (optional) if True, write start timestamps at the beginning of each segment
    """

    # convert to WAV
    wav_path = cached(
        file,
        ".wav",
        lambda output_filename: _run_ffmpeg(file, skip_seconds, length_seconds, output_filename),
        f"{skip_seconds}/{length_seconds}"
    )

    # run pyannote
    if detect_speakers:
        if not prepare_pyannote():
            raise ToolException("pyannote.audio initialization failed, speakers information not available")
        speakers_path = cached(
            wav_path,
            ".speakers.txt",
            lambda output_filename: diarize_and_write(wav_path, output_filename))
    else:
        speakers_path = None

    # run whisper
    whisper_args = []
    if prompt is None:
        prompt = "Split text into individual sentences. Use capital letters and punctuation."
    if len(prompt) > 0:
        whisper_args.append("--prompt")
        whisper_args.append(prompt)
    if extra_args is not None:
        whisper_args.extend(extra_args.split())
    raw_transcript_path = cached(
        wav_path,
        ".raw.txt",
        lambda output_filename: _run_whisper(wav_path, model, whisper_args, output_filename),
        discriminator=f"{model}/{whisper_args}")

    # read and merge segments
    if speakers_path is None:
        transcript_path = cached(
            raw_transcript_path,
            ".final.txt",
            lambda output_file: _build_final_result_no_speakers(raw_transcript_path, max_delay, max_length, output_file),
            discriminator=f"{max_delay}/{max_length}"
        )
    else:
        transcript_path = multi_cached(
            [raw_transcript_path, speakers_path],
            ".final.txt",
            lambda output_file: _build_final_result_with_speakers(raw_transcript_path, speakers_path, whisper_args, max_delay, max_length, write_timestamps, output_file),
            discriminator=f"{max_delay}/{max_length}/{write_timestamps}")

    # read final transcript
    with open(transcript_path, 'r') as file:
        transcript = file.read()
    return transcript


def _run_ffmpeg(input_file: str, skip_seconds: int, length_seconds: int, output_file: str):
    cmd = ["ffmpeg"]
    if skip_seconds is not None:
        cmd.extend(["-ss", str(skip_seconds)])
    cmd.extend([
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le"
    ])
    if length_seconds is not None:
        cmd.extend(["-t", str(length_seconds)])
    cmd.append(output_file)
    return run_process(cmd)


def _run_whisper(input_file: str, model: str, extra_args: [str], output_file: str):
    whisper_args = [
        "whisper-cli",
        "--model", model,
        "--file", input_file,
    ]
    whisper_args.extend(extra_args)
    segments = run_process(whisper_args)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(segments)


def _read_whisper_segments(file: str) -> List[Segment]:
    segments = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            segment = _convert_whisper_output_to_segment(line)
            if segment is not None:
                segments.append(segment)
    return segments

_whisper_segment_pattern = re.compile(
    r'^\[(\d+):(\d+):(\d+\.\d+)\s-->\s(\d+):(\d+):(\d+\.\d+)\]\s\s(.*)$'
)

def _convert_whisper_output_to_segment(line):
    match = _whisper_segment_pattern.match(line)
    if match:
        h1, m1, s1, h2, m2, s2, text = match.groups()
        start = float(h1)*3600 + float(m1)*60 + float(s1)
        end = float(h2)*3600 + float(m2)*60 + float(s2)
        if start == end and len(text) == 0:
            return None
        return Segment(start, end, text)
    return None


def _build_final_result_with_speakers(raw_transcript_path: str, speakers_path: str, whisper_args: List[str] | None, max_delay: float, max_length: int, write_timestamps: bool, output_path: str):
    whisper_segments = _read_whisper_segments(raw_transcript_path)
    speaker_segments = read_segments(speakers_path)
    # if extra_whisper_args contains "--max-len", then we need to merge sentences
    if "--max-len" in whisper_args:
        whisper_segments = build_phrases(whisper_segments, max_delay, max_length)
    whisper_segments = assign_speakers(whisper_segments, speaker_segments)
    whisper_segments = merge_segments(whisper_segments, max_delay, max_length, merge_sentences=True)
    for segment in whisper_segments:
        if segment.speaker is None:
            segment.speaker = "UNKNOWN"
    if write_timestamps:
        write_segments(whisper_segments, output_path, 'full')
    else:
        write_segments(whisper_segments, output_path, 'medium')


def _build_final_result_no_speakers(raw_transcript_path: str, max_delay: float, max_length: int, output_path: str):
    segments = _read_whisper_segments(raw_transcript_path)
    segments = merge_segments(segments, max_delay, max_length, merge_sentences=False)
    write_segments(segments, output_path, 'brief')


make_transcript = StructuredTool.from_function(
    _make_transcript,
    name="whisper_cpp",
    parse_docstring=True,
    error_on_invalid_docstring=True
)
