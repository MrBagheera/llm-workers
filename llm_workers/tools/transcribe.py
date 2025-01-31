import re
from typing import List, Literal

from marshmallow.fields import Boolean


class Segment:
    """Segment of text with start and end time and optional speakers names."""
    start: float
    end: float
    text: str
    speaker: str

    def __init__(self, start: float, end: float, text: str, speaker: str = None):
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker

    def __str__(self):
        speaker = "" if self.speaker is None else self.speaker
        return f"{self.start:.3f}-{self.end:.3f}:{speaker}:{self.text}"

    def __repr__(self):
        return self.__str__()

    _segment_pattern = re.compile(
        r'^(\d+)-(\d+):(.*)?:(.*)$'
    )

    @classmethod
    def read(cls, line):
        match = cls._segment_pattern.match(line)
        if not match:
            raise ValueError(f"Invalid segment line: {line}")
        start, end, speaker, text = match.groups()
        start = float(start)
        end = float(end)
        if len(speaker) == 0:
            speaker = None
        return cls(start, end, text, speaker)

    def copy(self):
        return Segment(self.start, self.end, self.text, self.speaker)

    def brief(self):
        return f"{self.text}\n"

    def full(self):
        if self.speaker is None:
            return f"{self.text}\n\n"
        return f"SPEAKER {self.speaker}:\n{self.text}\n\n"


# noinspection PyShadowingBuiltins
def write_segments(segments: List[Segment], type: Literal['raw', 'brief', 'full'], filename):
    with open(filename, "w", encoding="utf-8") as f:
        for segment in segments:
            if type == 'raw':
                f.write(str(segment))
            elif type == 'brief':
                f.write(segment.brief())
            else:
                f.write(segment.full())

def read_segments(filename) -> List[Segment]:
    with open(filename, "r", encoding="utf-8") as f:
        segments = []
        for line in f:
            if len(line.strip()) > 0:
                segments.append(Segment.read(line))
        return segments

def merge_segments(segments: List[Segment], max_delay: float, max_length: int, merge_sentences: bool) -> List[Segment]:
    if len(segments) == 0:
        return segments
    last = segments[0].copy()
    merged = [last]
    for segment in segments[1:]:
        segment_len = len(segment.text)
        if last.speaker != segment.speaker:
            can_merge = False
        elif segment_len > max_length:
            can_merge = False
        elif segment.start - last.end > max_delay:
            can_merge = False
        else:
            can_merge = merge_sentences or not last.text.endswith(('.', ',', '!', '?'))
        if can_merge:
            last.end = segment.end
            if not last.text.endswith(' ') and not segment.text.startswith(('.', ',', '!', '?')):
                last.text += " "
            last.text += segment.text.strip()
        else:
            last = segment.copy()
            last.text = last.text.strip()
            merged.append(last)
    return merged
