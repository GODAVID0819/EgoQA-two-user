from __future__ import annotations

import copy
import unittest

from summarize_preprocessing_throughput_sweep import (
    validate_full_unpruned_packets,
)
from ten_minute_six_user_setup import representative_six_user_packet


class PreprocessingThroughputSummaryTests(unittest.TestCase):
    def test_full_unpruned_representative_packet_passes(self) -> None:
        result = validate_full_unpruned_packets(
            [representative_six_user_packet()]
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["packet_frame_counts"], [18])

    def test_missing_provider_frame_fails_contract(self) -> None:
        packet = copy.deepcopy(representative_six_user_packet())
        packet["clips"][1]["frames"].pop()

        result = validate_full_unpruned_packets([packet])

        self.assertFalse(result["passed"])
        self.assertGreater(result["error_count"], 0)
        self.assertTrue(
            any("count mismatch" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
