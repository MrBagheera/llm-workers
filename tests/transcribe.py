import unittest
from llm_workers.tools.transcribe_common import Segment, merge_segments


class TestSegment(unittest.TestCase):

    def test_segment_initialization(self):
        segment = Segment(0.0, 1.0, "Hello world", "A")
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 1.0)
        self.assertEqual(segment.text, "Hello world")
        self.assertEqual(segment.speaker, "A")
        segment = Segment(0.0, 1.0, "Hello world")
        self.assertEqual(segment.speaker, "")

    def test_segment_write(self):
        segment = Segment(0.0, 1.0, "Hello world", "A")
        self.assertEqual(str(segment), "0.000-1.000:A:Hello world")
        segment = Segment(0.0, 1.0, "Hello world", "")
        self.assertEqual(str(segment), "0.000-1.000::Hello world")

    def test_segment_read_valid(self):
        line = "0-1:A:Hello world"
        segment = Segment.read(line)
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 1.0)
        self.assertEqual(segment.text, "Hello world")
        self.assertEqual(segment.speaker, "A")

    def test_segment_read_invalid(self):
        line = "invalid line"
        with self.assertRaises(ValueError):
            Segment.read(line)

    def test_segment_format_no_speaker(self):
        segment = Segment(0.0, 1.0, "Hello world")
        self.assertEqual(segment.format(), "Hello world\n\n")

    def test_segment_format_single_speaker(self):
        segment = Segment(0.0, 1.0, "Hello world", "A")
        self.assertEqual(segment.format(), "SPEAKER A:\nHello world\n\n")

    def test_segment_format_multiple_speakers(self):
        segment = Segment(0.0, 1.0, "Hello world", "A and B")
        self.assertEqual(segment.format(), "SPEAKERS A and B:\nHello world\n\n")

class TestSegmentsMerge(unittest.TestCase):

    def test_segments_merge(self):
        segments = [
            Segment(0.0, 1.0, "Hello", "A"),
            Segment(1.0, 2.0, " world", "A"),
            Segment(2.0, 3.0, "!", "A"),
            Segment(5.0, 6.0, " How", "A"),
            Segment(6.0, 7.0, " are", "A"),
            Segment(7.0, 8.0, " you", "A"),
            Segment(8.0, 9.0, "?", "A"),
            Segment(10.0, 11.0, "I", "B"),
            Segment(11.0, 12.0, " am", "B"),
            Segment(12.0, 13.0, " perfectly", "B"),
            Segment(13.0, 14.0, " fine", "B"),
            Segment(14.0, 15.0, ".", "B"),
        ]
        self.assertEqual(
            [
                "0.000-9.000:A:Hello world! How are you?",
                "10.000-15.000:B:I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 2.5, 100)]
        )
        self.assertEqual(
            [
                "0.000-3.000:A:Hello world!",
                "5.000-9.000:A:How are you?",
                "10.000-15.000:B:I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 1.5, 100)]
        )
        self.assertEqual(
            [
                "0.000-9.000:A:Hello world! How are you?",
                "10.000-15.000:B:I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 10.0, 15)]
        )

if __name__ == '__main__':
    unittest.main()