import unittest
from llm_workers.tools.transcribe import Segment, merge_segments


class TestSegment(unittest.TestCase):

    def test_segment_initialization(self):
        segment = Segment(0.0, 1.0, "Hello world", "A")
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 1.0)
        self.assertEqual(segment.text, "Hello world")
        self.assertEqual(segment.speaker, "A")
        segment = Segment(0.0, 1.0, "Hello world")
        self.assertEqual(segment.speaker, None)

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
        self.assertEqual(segment.txt(), "Hello world\n\n")

    def test_segment_format_with_speaker(self):
        segment = Segment(0.0, 1.0, "Hello world", "A")
        self.assertEqual(segment.txt(), "SPEAKER A:\nHello world\n\n")


class TestSegmentsMerge(unittest.TestCase):

    def test_segments_merge_for_no_diarization(self):
        segments = [
            Segment(0.0, 3.0, "Hello world! How"),
            Segment(3.5, 9.0, " are you?"),
            Segment(10.0, 15.0, "I am perfectly fine."),
        ]
        self.assertEqual(
            [
                "0.000-9.000::Hello world! How are you?",
                "10.000-15.000::I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 1, 100, False)]
        )

    def test_segments_merge_for_diarization1(self):
        segments = [
            Segment(0.0, 1.0, "Hello"),
            Segment(1.0, 2.0, " world"),
            Segment(2.0, 3.0, "!"),
            Segment(4.0, 5.0, " How"),
            Segment(6.0, 7.0, " are"),
            Segment(7.0, 8.0, " you"),
            Segment(8.0, 9.0, "?"),
            Segment(10.0, 11.0, "I"),
            Segment(11.0, 12.0, " am"),
            Segment(12.0, 13.0, " perfectly"),
            Segment(13.0, 14.0, " fine"),
            Segment(14.0, 15.0, "."),
        ]
        self.assertEqual(
            [
                "0.000-3.000::Hello world!",
                "4.000-9.000::How are you?",
                "10.000-15.000::I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 1.5, 100, False)]
        )
        self.assertEqual(
            [
                "0.000-3.000::Hello world!",
                "4.000-5.000::How",
                "6.000-9.000::are you?",
                "10.000-15.000::I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 0.5, 100, False)]
        )

    def test_segments_merge_for_diarization2(self):
        segments = [
            Segment(0.0, 3.0, "Hello world!", "A"),
            Segment(4, 9.0, "How are you?", "A"),
            Segment(10.0, 15.0, "I am perfectly fine.", "B"),
        ]
        self.assertEqual(
            [
                "0.000-3.000:A:Hello world!",
                "4.000-9.000:A:How are you?",
                "10.000-15.000:B:I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 0.5, 100, True)]
        )
        self.assertEqual(
            [
                "0.000-9.000:A:Hello world! How are you?",
                "10.000-15.000:B:I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 1.5, 100, True)]
        )
        self.assertEqual(
            [
                "0.000-3.000:A:Hello world!",
                "4.000-9.000:A:How are you?",
                "10.000-15.000:B:I am perfectly fine.",
            ],
            [str(s) for s in merge_segments(segments, 10, 10, True)]
        )

if __name__ == '__main__':
    unittest.main()