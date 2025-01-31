import logging
import re
from re import Pattern
from typing import List

from llm_workers.tools.transcribe import Segment, write_segments
from llm_workers.tools.utils import get_environment_variable

logger =  logging.getLogger(__name__)

_pipeline: callable = None
_pipeline_failed: bool = False

def is_pyannote_initialized():
    global _pipeline
    return _pipeline is not None

def prepare_pyannote(device: str = "mps"):
    global _pipeline
    global _pipeline_failed
    if _pipeline_failed:
        return False
    if _pipeline is not None:
        return True
    try:
        from pyannote.audio import Pipeline
        import torch
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=get_environment_variable("HF_TOKEN", default=None))
        _pipeline.to(torch.device(device))
        return True
    except Exception as e:
        logger.warning(f"pyannote.audio initialization failed, speakers information not available. Cause: {e}")
        _pipeline_failed = True
        _pipeline = None
        return False


def diarize(filename: str) -> List[Segment]:
    if not is_pyannote_initialized():
        return []
    global _pipeline
    diarization = _pipeline(filename)
    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append(Segment(
            start=turn.start,
            end=turn.end,
            text='',
            speaker=_resolve_speaker(speaker)))
    return result

_pyannote_speaker_pattern: Pattern[str] = re.compile(
    r'^SPEAKER_(\d+)$'
)

def _resolve_speaker(speaker: str) -> str | None:
    # if SPEAKER_XX, return XX
    match = _pyannote_speaker_pattern.match(speaker)
    if match:
        # convert match.group(1) to int
        idx = int(match.group(1))
        # convert idx to character
        return chr(ord('A') + idx)
    else:
        return None

# def _clean_diarization_results(diarize_results: List[Segment], min_duration: float):
#     result = []
#     for s in diarize_results:
#         is_too_short = s.end - s.start < min_duration
#         if is_too_short:
#             continue
#         result.append(s)
#     return result

def diarize_and_write(filename: str, output_filename: str):
    segments = diarize(filename)
    write_segments(segments, output_filename, type='raw')