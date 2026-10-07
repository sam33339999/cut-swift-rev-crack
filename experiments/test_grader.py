"""Drive the shipped grader. These cases are the visible-reply rule, not a second implementation."""

import unittest

from experiments.grader import extract_answer, is_correct


class GraderTest(unittest.TestCase):
    def test_matching_answer_line(self):
        content = "The product is ready.\nANSWER: 391"
        self.assertEqual(extract_answer(content), "391")
        self.assertTrue(is_correct(content, "391", truncated=False))

    def test_comma_formatted_integer(self):
        content = "ANSWER: 31,875,000"
        self.assertEqual(extract_answer(content), "31875000")
        self.assertTrue(is_correct(content, "31875000", truncated=False))

    def test_extra_zero_is_wrong(self):
        content = "ANSWER: 318750000"
        self.assertEqual(extract_answer(content), "318750000")
        self.assertFalse(is_correct(content, "31875000", truncated=False))

    def test_missing_answer_line(self):
        content = "The answer is 391 but it is not on an ANSWER line."
        self.assertIsNone(extract_answer(content))
        self.assertFalse(is_correct(content, "391", truncated=False))

    def test_truncated_flag_overrides_a_matching_line(self):
        content = "ANSWER: 615"
        self.assertEqual(extract_answer(content), "615")
        self.assertFalse(is_correct(content, "615", truncated=True))

    def test_final_answer_line_wins(self):
        content = "ANSWER: 1\nANSWER: 405\n"
        self.assertEqual(extract_answer(content), "405")
        self.assertTrue(is_correct(content, "405", truncated=False))


if __name__ == "__main__":
    unittest.main()
