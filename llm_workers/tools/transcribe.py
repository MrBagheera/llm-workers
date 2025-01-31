import re
from typing import List, Literal
from collections import defaultdict


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
        r'^(\d+\.\d+)-(\d+\.\d+):(.*)?:(.*)$'
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
        return f"{self.text}"

    def full(self):
        if self.speaker is None:
            return f"{self.text}\n"
        return f"SPEAKER {self.speaker}:\n{self.text}\n"


# noinspection PyShadowingBuiltins
def write_segments(segments: List[Segment], output_filename: str, type: Literal['raw', 'brief', 'full']):
    with open(output_filename, "w", encoding="utf-8") as f:
        for segment in segments:
            if type == 'raw':
                f.write(str(segment))
            elif type == 'brief':
                f.write(segment.brief())
            else:
                f.write(segment.full())
            f.write('\n')


def read_segments(filename) -> List[Segment]:
    with open(filename, "r", encoding="utf-8") as f:
        segments = []
        for line in f:
            if len(line.strip()) > 0:
                segments.append(Segment.read(line))
        return segments


def build_phrases(segments: List[Segment], max_delay: float, max_length: int) -> List[Segment]:
    if len(segments) == 0:
        return segments
    last = segments[0].copy()
    merged = [last]
    for segment in segments[1:]:
        last_len = len(last.text)
        segment_len = len(segment.text)
        last_segment_complete = last.text.endswith(('.', '!', '?'))
        if last.speaker != segment.speaker:
            can_merge = False
        elif last_len + segment_len > max_length:
            can_merge = False
        elif segment.start - last.end > max_delay:
            can_merge = False
        else:
            if last_segment_complete:
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


def merge_segments(segments: List[Segment], max_delay: float, max_length: int, merge_sentences: bool) -> List[Segment]:
    if len(segments) == 0:
        return segments
    last = segments[0].copy()
    merged = [last]
    for segment in segments[1:]:
        last_len = len(last.text)
        segment_len = len(segment.text)
        last_segment_complete = last.text.endswith(('.', '!', '?'))
        if last.speaker != segment.speaker:
            can_merge = False
        elif last_len + segment_len > max_length:
            can_merge = False
        elif segment.start - last.end > max_delay:
            can_merge = False
        else:
            if last_segment_complete and not merge_sentences:
                can_merge = False
            else:
                can_merge = True
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


def assign_speakers(transcribe_results: List[Segment], speaker_segments: List[Segment]) -> List[Segment]:
    assigned = []
    i = 0  # pointer into speaker_segments
    for t in transcribe_results:
        overlap_by_speaker = defaultdict(float)
        # Skip past any speaker segments that end before t starts
        while i < len(speaker_segments) and speaker_segments[i].end < t.start:
            i += 1
        # Accumulate overlap for all speaker segments that might intersect
        j = i # separate because we want to keep i
        while j < len(speaker_segments) and speaker_segments[j].start < t.end:
            s = speaker_segments[j]
            if s.speaker is not None:
                overlap = min(t.end, s.end) - max(t.start, s.start)
                if overlap > 0:
                    overlap_by_speaker[s.speaker] += overlap
            j += 1
        # Assign the speaker with the largest total overlap
        if overlap_by_speaker:
            t = t.copy()
            t.speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
        assigned.append(t)
    return assigned
