"""
Boss Strategies — Training/Test Configuration

Based on Strong Trend periods, extended:
- Training: ±2 months on each end
- Test: +3 months before start
"""

# Training: strong_trend periods extended ±2 months
TRAINING_SYMBOLS = ["J", "ZC", "JM", "I", "NI", "SA"]
TRAINING_PERIODS = {
    "J":  ("2015-04-01", "2017-08-01"),
    "ZC": ("2020-04-01", "2022-08-01"),
    "JM": ("2020-04-01", "2022-08-01"),
    "I":  ("2015-04-01", "2017-08-01"),
    "NI": ("2020-11-01", "2022-11-01"),
    "SA": ("2022-04-01", "2024-08-01"),
}

# Test: strong_trend test periods with 3 months earlier start
TEST_PERIODS = {
    "AG": ("2024-10-01", "2026-03-01"),
    "EC": ("2023-04-01", "2024-09-01"),
}
