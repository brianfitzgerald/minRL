from minrl.utils import clean_observation


def test_clean_observation_basic():
    obs = "  Hello world! Score: 100 Moves: 50  "
    expected = "Hello world!"
    assert clean_observation(obs) == expected


def test_clean_observation_with_caret():
    obs = "Line 1\n^Line 2\nLine 3"
    expected = "Line 1"
    assert clean_observation(obs) == expected


def test_clean_observation_with_gt_and_purpose():
    obs = "Line 1\n> Line 2\nLine 3 Purpose: some purpose\nLine 4"
    expected = "Line 1\nLine 4"
    assert clean_observation(obs) == expected


def test_clean_observation_empty_string():
    obs = ""
    expected = ""
    assert clean_observation(obs) == expected


def test_clean_observation_only_ignored_lines():
    obs = "> Line 1\nPurpose: some purpose\n^Line 3"
    expected = ""
    assert clean_observation(obs) == expected


def test_clean_observation_complex_case():
    obs = """
    You are in a dark room. Score: 10 Moves: 20
    > You see a key.
    This is another line.
    Purpose: to test
    ^This line should be ignored.
    Final line.
    """
    expected = "You are in a dark room.\nThis is another line."
    assert clean_observation(obs) == expected


def test_clean_observation_multiple_score_moves():
    obs = "First line Score: 1 Moves: 2\nSecond line Score: 3 Moves: 4"
    expected = "First line\nSecond line"
    assert clean_observation(obs) == expected


def test_clean_observation_no_score_moves():
    obs = "Just a regular line.\nAnother regular line."
    expected = "Just a regular line.\nAnother regular line."
    assert clean_observation(obs) == expected


def test_clean_observation_leading_trailing_whitespace():
    obs = "  leading and trailing  "
    expected = "leading and trailing"
    assert clean_observation(obs) == expected


def test_clean_observation_only_caret():
    obs = "^"
    expected = ""
    assert clean_observation(obs) == expected


def test_clean_observation_caret_at_start_of_line_with_whitespace():
    obs = "Line 1\n  ^Line 2\nLine 3"
    expected = "Line 1"
    assert clean_observation(obs) == expected


def test_clean_observation_purpose_with_whitespace():
    obs = "Line 1\n  Purpose: some purpose  \nLine 2"
    expected = "Line 1\nLine 2"
    assert clean_observation(obs) == expected


def test_clean_observation_gt_with_whitespace():
    obs = "Line 1\n  > Line 2  \nLine 3"
    expected = "Line 1\nLine 3"
    assert clean_observation(obs) == expected
