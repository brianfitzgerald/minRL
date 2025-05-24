from tasks.connections import parse_groups, score_connections_hard


FIRST_SAMPLE = {
    "groups": [
        {"reason": "defeat badly", "words": ["crush", "rout", "shellac", "trash"]},
        {"reason": "tops", "words": ["cami", "halter", "tank", "tee"]},
        {"reason": "butt", "words": ["bottom", "buns", "seat", "tail"]},
        {"reason": "kinds of whales", "words": ["blue", "fin", "gray", "right"]},
    ],
    "words": [
        "crush",
        "rout",
        "shellac",
        "trash",
        "cami",
        "halter",
        "tank",
        "tee",
        "bottom",
        "buns",
        "seat",
        "tail",
        "blue",
        "fin",
        "gray",
        "right",
    ],
    "prompt": [
        {
            "content": "\nYou are an expert puzzle solving model.\nFind groups of words that are related to each other. Each group is four words long. There are exactly four groups in total.\nYou may only use each word in one group.\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n<group>\n...\n</group>\n<group>\n...\n</group>\n</answer>\n\n# Example\n\nUser: apple, orange, banana, pear, corolla, charger,\nAssistant: <reasoning>\nThe first group are all fruits.\nThe second group are all cars.\n</reasoning>\n<answer>\n<group>apple, orange, banana, pear</group>\n<group>corolla, charger</group>\n</answer>\n\n# Example\n\nUser: dog, cat, red, white,\nAssistant: <reasoning>\nThe first group are all animals.\nThe second group are all colors.\n</reasoning>\n<answer>\n<group>dog, cat</group>\n<group>red, white</group>\n</answer>\n",
            "role": "system",
        },
        {
            "content": "crush, rout, shellac, trash, cami, halter, tank, tee, bottom, buns, seat, tail, blue, fin, gray, right",
            "role": "user",
        },
    ],
    "answer": "<answer><group>crush, rout, shellac, trash</group>\n<group>cami, halter, tank, tee</group>\n<group>bottom, buns, seat, tail</group>\n<group>blue, fin, gray, right</group></answer>",
    "answer_groups": [
        ["crush", "rout", "shellac", "trash"],
        ["cami", "halter", "tank", "tee"],
        ["bottom", "buns", "seat", "tail"],
        ["blue", "fin", "gray", "right"],
    ],
}

ANSWERS_BATCH = [FIRST_SAMPLE["answer_groups"]]


def test_parse_groups_single():
    input_str = "<group>apple, banana, cherry</group>"
    expected = [["apple", "banana", "cherry"]]
    assert parse_groups(input_str) == expected


def test_parse_groups_multiple():
    input_str = "<group>apple, banana</group> some text <group>cherry, date</group>"
    expected = [["apple", "banana"], ["cherry", "date"]]
    assert parse_groups(input_str) == expected


def test_parse_groups_whitespace():
    input_str = "<group>  apple  ,  banana  </group>"
    expected = [["apple", "banana"]]
    assert parse_groups(input_str) == expected


def test_parseG_groups_empty_string():
    input_str = ""
    expected = []
    assert parse_groups(input_str) == expected


def test_parse_groups_empty_group():
    input_str = "<group></group>"
    expected = [[]]
    assert parse_groups(input_str) == expected


def test_uppercase():
    input_str = "<think>\nFirst, identify groups of four words related by category or characteristics.\n\nGroup 1: Weather types - HAIL, RAIN, SLEET, SNOW (all types of precipitation)\nGroup 2: Palindromes - KAYAK, LEVEL, MOM, RACECAR (words spelled the same forward and backward)\nGroup 3: Basketball teams or related terms - BUCKS, HEAT, JAZZ, NETS (NBA teams)\nGroup 4: Keyboard-related terms - OPTION, RETURN, SHIFT, TAB (all keyboard keys/functions)\n</think>\n<answer>\n<group> HAIL, RAIN, SLEET, SNOW </group>\n<group> BUCKS, HEAT, JAZZ, NETS </group>\n<group> OPTION, RETURN, SHIFT, TAB </group>\n<group> KAYAK, LEVEL, MOM, RACECAR </group>\n</answer>"
    groups = parse_groups(input_str)
    score = score_connections_hard(
        [
            ["HAIL", "RAIN", "SLEET", "SNOW"],
            ["BOOT", "LOAFER", "PUMP", "SNEAKER"],
            ["CHEEK", "EYE", "MOUTH", "NOSE"],
            ["ADIDAS", "NIKE", "PUMA", "REEBOK"],
        ],
        groups,
    )
    assert score == 0.25
