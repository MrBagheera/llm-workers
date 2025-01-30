import re
from typing import List

class Segment:
    """Segment of text with start and end time and optional speaker name."""
    start: float
    end: float
    text: str
    speaker: str

    def __init__(self, start: float, end: float, text: str, speaker: str = ''):
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker

    def __str__(self):
        return f"{self.start:.3f}-{self.end:.3f}:{self.speaker}:{self.text}"

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
        return cls(start, end, text, speaker)

    def copy(self):
        return Segment(self.start, self.end, self.text, self.speaker)

    def format(self):
        if len(self.speaker) == 0:
            return f"{self.text}\n\n"
        if len(self.speaker) == 1:
            return f"SPEAKER {self.speaker}:\n{self.text}\n\n"
        else:
            return f"SPEAKERS {self.speaker}:\n{self.text}\n\n"


def merge_segments(segments: List[Segment], max_delay: float, max_length: int) -> List[Segment]:
    if len(segments) == 0:
        return segments
    hard_max_length = max_length * 2
    last = segments[0].copy()
    merged = [last]
    for segment in segments[1:]:
        segment_len = len(segment.text)
        if last.speaker != segment.speaker:
            can_merge = False
        elif segment_len > hard_max_length:
            can_merge = False
        elif segment_len > max_length and last.text[-1] in '.!?':
            can_merge = False
        elif segment.start - last.end > max_delay:
            can_merge = False
        else:
            can_merge = True
        if can_merge:
            last.end = segment.end
            last.text += segment.text
        else:
            last = segment.copy()
            last.text = last.text.lstrip()
            merged.append(last)
    return merged


