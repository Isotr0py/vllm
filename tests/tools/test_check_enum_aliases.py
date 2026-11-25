# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from textwrap import dedent

import pytest

from tools.pre_commit.check_enum_aliases import check_source


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ('"value"', '"value"'),
        ("None", "None"),
        ("True", "1"),
        ("(1, 2)", "(1, 2)"),
        ("-1", "~0"),
        ("(1 + 2) * 4", "24 >> 1"),
    ],
)
def test_literal_and_arithmetic_duplicates(first, second):
    source = f"class Sample(Enum):\n    FIRST = {first}\n    SECOND = {second}\n"

    errors = check_source(source, "sample.py")

    assert len(errors) == 1
    assert errors[0].startswith("sample.py:3:")
    assert "'SECOND'" in errors[0]
    assert "'FIRST'" in errors[0]


def test_member_references_and_diagnostic_lines():
    source = dedent(
        """
        class Sample(enum.Enum):
            FIRST = 1
            ALIAS = FIRST
            PAIR = (FIRST, 2)
            PAIR_AGAIN = (1, 2)
            NEGATIVE = -1
            NEGATIVE_AGAIN = ~0
        """
    )

    errors = check_source(source, "enums.py")

    assert [error.split(":", 2)[:2] for error in errors] == [
        ["enums.py", "4"],
        ["enums.py", "6"],
        ["enums.py", "8"],
    ]
    assert "same value as 'FIRST'" in errors[0]
    assert "same value as 'PAIR'" in errors[1]
    assert "same value as 'NEGATIVE'" in errors[2]


@pytest.mark.parametrize(
    "bases",
    ["Enum", "enum.Enum", "IntEnum", "enum.IntEnum", "StrEnum", "str, Enum"],
)
def test_direct_enum_bases_are_checked(bases):
    source = f"class Sample({bases}):\n    FIRST = 1\n    SECOND = 1\n"
    assert len(check_source(source)) == 1


@pytest.mark.parametrize(
    "bases", ["Flag", "IntFlag", "enum.Flag", "BaseEnum", "object"]
)
def test_non_target_classes_are_ignored(bases):
    source = f"class Sample({bases}):\n    FIRST = 1\n    SECOND = 1\n"
    assert check_source(source) == []


def test_int_enum_auto_values_and_tuple_auto():
    source = dedent(
        """
        class Sample(IntEnum):
            FIRST = auto()
            SECOND = enum.auto()
            SECOND_AGAIN = SECOND
            EXPLICIT = 5
            SIX = auto()
            SIX_AGAIN = 6
            PAIR = (10, auto())
            EIGHT = auto()
            EIGHT_AGAIN = 8
        """
    )

    errors = check_source(source, "auto.py")

    assert [error.split(":", 2)[1] for error in errors] == ["5", "8", "11"]
    assert "same value as 'SECOND'" in errors[0]
    assert "same value as 'SIX'" in errors[1]
    assert "same value as 'EIGHT'" in errors[2]


def test_auto_falls_back_to_start_when_values_cannot_increment():
    source = (
        'class Sample(Enum):\n    FIRST = "first"\n    SECOND = auto()\n    THIRD = 1\n'
    )

    assert check_source(source, "auto_start.py") == [
        "auto_start.py:4: duplicate enum member 'THIRD' has the same value as 'SECOND'"
    ]


def test_auto_uses_largest_sortable_value():
    source = (
        "class Sample(IntEnum):\n"
        "    FIRST = 2\n"
        "    SECOND = 1\n"
        "    THIRD = auto()\n"
        "    FOURTH = 2\n"
    )

    assert check_source(source, "auto_max.py") == [
        "auto_max.py:5: duplicate enum member 'FOURTH' has the same value as 'FIRST'"
    ]


def test_auto_reverse_finds_last_incrementable_value():
    source = (
        "class Sample(Enum):\n"
        "    FIRST = 1\n"
        '    SECOND = "second"\n'
        "    THIRD = auto()\n"
        "    FOURTH = 2\n"
    )

    assert check_source(source, "auto_reverse.py") == [
        (
            "auto_reverse.py:5: duplicate enum member 'FOURTH' has the same value as "
            "'THIRD'"
        )
    ]


def test_str_enum_auto_uses_lowercase_member_name():
    source = dedent(
        """
        class Sample(StrEnum):
            FIRST_NAME = auto()
            SECOND = enum.auto()
            FIRST_AGAIN = "first_name"
            SECOND_AGAIN = "second"
        """
    )

    errors = check_source(source)

    assert len(errors) == 2
    assert "same value as 'FIRST_NAME'" in errors[0]
    assert "same value as 'SECOND'" in errors[1]


def test_unknown_expressions_do_not_create_guessed_aliases():
    source = dedent(
        """
        class Sample(Enum):
            FIRST = dynamic()
            REFERENCE = FIRST
            TUPLE = (REFERENCE, 1)
            LIST = [FIRST]
            MAPPING = {FIRST: 1}
            NEXT = auto()
            SAFE = 1 + 2
            SAFE_AGAIN = 3
        """
    )

    errors = check_source(source)

    assert len(errors) == 1
    assert "same value as 'SAFE'" in errors[0]


def test_reasoned_alias_exemption():
    source = dedent(
        """
        class Sample(Enum):
            FIRST = 1
            COMPAT_ALIAS = 1  # enum-alias: ok (kept for API compatibility)
        """
    )
    assert check_source(source) == []


@pytest.mark.parametrize(
    "comment",
    [
        "# enum-alias: ok",
        "# enum-alias: ok ()",
        "# enum-alias: ok (   )",
        "# enum-alias: ok compatibility",
    ],
)
def test_alias_exemption_requires_a_nonempty_reason(comment):
    source = f"class Sample(Enum):\n    FIRST = 1\n    SECOND = 1  {comment}\n"
    assert len(check_source(source)) == 1


def test_exemption_must_be_on_the_member_line():
    source = dedent(
        """
        class Sample(Enum):
            FIRST = 1
            # enum-alias: ok (explained on the wrong line)
            SECOND = 1
        """
    )
    assert len(check_source(source)) == 1


def test_annotated_members_and_syntax_errors():
    annotated = "class Sample(Enum):\n    FIRST: int = 1\n    SECOND: int = 1\n"
    assert len(check_source(annotated)) == 1
    assert check_source("class Sample(Enum:\n") == []
