from __future__ import annotations

import unittest

from training.grpo_v3.shared.json_format import validate_completion_json


class JsonFormatValidationTests(unittest.TestCase):
    def test_raw_object_is_valid_without_repair(self) -> None:
        raw = '  {"a": 1, "message": "ok"}\n'
        result = validate_completion_json(raw)

        self.assertEqual(result.status, "raw_valid")
        self.assertEqual(result.value, {"a": 1, "message": "ok"})
        self.assertEqual(result.raw_completion, raw)
        self.assertIsNone(result.repaired_completion)
        self.assertEqual(result.repair_operations, [])
        self.assertEqual(result.format_penalty, 0.0)
        self.assertFalse(result.semantic_text_changed)

    def test_markdown_fence_is_repaired_and_audited(self) -> None:
        raw = '```json\n{"a": 1}\n```'
        result = validate_completion_json(raw)

        self.assertEqual(result.status, "repaired")
        self.assertEqual(result.value, {"a": 1})
        self.assertEqual(result.repaired_completion, '{"a": 1}')
        self.assertEqual(
            result.repair_operations,
            [{"operation": "strip_markdown_fence", "position": 0}],
        )
        self.assertEqual(result.format_penalty, -0.5)

    def test_missing_object_member_comma_is_inserted_at_original_position(self) -> None:
        raw = '{"combined_answerability":"sufficient"\n"generator_rationale":"ok"}'
        result = validate_completion_json(raw)

        self.assertEqual(result.status, "repaired")
        self.assertEqual(result.value["generator_rationale"], "ok")
        self.assertEqual(
            result.repair_operations,
            [{
                "operation": "insert_missing_member_comma",
                "position": raw.index('\n"generator') + 1,
            }],
        )

    def test_trailing_object_and_array_commas_are_removed(self) -> None:
        raw = '{"a": [1,],}'
        result = validate_completion_json(raw)

        self.assertEqual(result.status, "repaired")
        self.assertEqual(result.value, {"a": [1]})
        self.assertEqual(
            [operation["operation"] for operation in result.repair_operations],
            ["remove_trailing_comma", "remove_trailing_comma"],
        )
        self.assertEqual(
            [operation["position"] for operation in result.repair_operations],
            [raw.index(",]"), raw.rindex(",}")],
        )

    def test_punctuation_like_content_inside_strings_is_never_changed(self) -> None:
        raw = '{"message":"literal }\\n\\\"key\\\" and [,] stays",\n"a":1}'
        result = validate_completion_json(raw)

        self.assertEqual(result.status, "raw_valid")
        self.assertEqual(result.value["message"], 'literal }\n"key" and [,] stays')
        self.assertEqual(result.repair_operations, [])

    def test_escaped_quotes_and_backslashes_do_not_break_scanner_state(self) -> None:
        raw = '{"path":"C:\\\\tmp\\\\file", "quote":"say \\\"hello\\\""\n"a":1}'
        result = validate_completion_json(raw)

        self.assertEqual(result.status, "repaired")
        self.assertEqual(result.value["path"], "C:\\tmp\\file")
        self.assertEqual(result.value["quote"], 'say "hello"')
        self.assertEqual(result.repair_operations[0]["operation"], "insert_missing_member_comma")

    def test_truncated_json_is_unrecoverable_with_parser_location(self) -> None:
        raw = '{"a": 1, "b": [2, 3]'
        result = validate_completion_json(raw)

        self.assertEqual(result.status, "unrecoverable")
        self.assertIsNone(result.value)
        self.assertIsNone(result.repaired_completion)
        self.assertEqual(result.format_penalty, -3.0)
        self.assertEqual(result.parse_error["type"], "JSONDecodeError")
        self.assertIn("lineno", result.parse_error)
        self.assertIn("colno", result.parse_error)
        self.assertIn("pos", result.parse_error)

    def test_unclosed_string_is_unrecoverable(self) -> None:
        result = validate_completion_json('{"a": "unterminated}')

        self.assertEqual(result.status, "unrecoverable")
        self.assertEqual(result.repair_operations, [])

    def test_more_than_three_repairs_is_unrecoverable(self) -> None:
        result = validate_completion_json('{"a":[1,],"b":[2,],"c":[3,],"d":[4,]}')

        self.assertEqual(result.status, "unrecoverable")
        self.assertIsNone(result.repaired_completion)
        self.assertEqual(result.repair_operations, [])
        self.assertEqual(result.parse_error["reason"], "repair_operation_limit_exceeded")

    def test_array_member_missing_comma_is_not_repaired(self) -> None:
        result = validate_completion_json('{"a": [1 2]}')

        self.assertEqual(result.status, "unrecoverable")
        self.assertEqual(result.repair_operations, [])

    def test_non_object_json_is_unrecoverable(self) -> None:
        result = validate_completion_json('[{"a": 1}]')

        self.assertEqual(result.status, "unrecoverable")
        self.assertEqual(result.parse_error["reason"], "top_level_json_must_be_object")


if __name__ == "__main__":
    unittest.main()
