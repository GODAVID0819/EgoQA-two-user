import pytest

from egolife_two_user_qa.schema import extract_json_object


def test_extract_json_object_uses_last_complete_object_after_reasoning() -> None:
    raw = (
        '<think>Compare {A, B}; an example is {"draft": true}.</think>\n'
        '{"reason": "supported", "needed_facts": []}'
    )

    assert extract_json_object(raw) == {
        "reason": "supported",
        "needed_facts": [],
    }


def test_extract_json_object_uses_final_object_after_example_object() -> None:
    raw = (
        'Example: {"status": "FAIL"}\n'
        'Final: {"status": "PASS", "reason": "visible"}'
    )

    assert extract_json_object(raw) == {
        "status": "PASS",
        "reason": "visible",
    }


def test_extract_json_object_rejects_truncated_final_object() -> None:
    with pytest.raises(ValueError, match="No complete JSON object"):
        extract_json_object('<think>done</think>\n{"status": "PASS"')
