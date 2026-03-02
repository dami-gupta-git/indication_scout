"""Unit tests for indication_scout.markers."""

from typing import Annotated, get_type_hints

import pytest

from indication_scout.markers import _NoReview, is_no_review, no_review


# ---------------------------------------------------------------------------
# Decorator: functions
# ---------------------------------------------------------------------------


def test_no_review_sets_attribute_on_function():
    @no_review
    def helper():
        pass

    assert helper.__no_review__ is True


def test_no_review_returns_original_function():
    def helper():
        return 42

    decorated = no_review(helper)
    assert decorated is helper
    assert decorated() == 42


# ---------------------------------------------------------------------------
# Decorator: classes
# ---------------------------------------------------------------------------


def test_no_review_sets_attribute_on_class():
    @no_review
    class Legacy:
        pass

    assert Legacy.__no_review__ is True


def test_no_review_returns_original_class():
    @no_review
    class Legacy:
        value = 7

    assert Legacy.value == 7


# ---------------------------------------------------------------------------
# Decorator: methods
# ---------------------------------------------------------------------------


def test_no_review_sets_attribute_on_method():
    class MyClass:
        @no_review
        def compute(self):
            return "result"

    assert MyClass.compute.__no_review__ is True
    assert MyClass().compute() == "result"


# ---------------------------------------------------------------------------
# Annotated field metadata
# ---------------------------------------------------------------------------


def test_no_review_usable_as_annotated_metadata():
    class MyModel:
        internal_id: Annotated[str, no_review]

    hints = get_type_hints(MyModel, include_extras=True)
    metadata = hints["internal_id"].__metadata__
    assert no_review in metadata


def test_is_no_review_true_for_annotated_metadata_sentinel():
    # The sentinel itself must carry the marker so the agent can detect it
    # inside Annotated metadata without special-casing the type.
    assert is_no_review(no_review) is True


# ---------------------------------------------------------------------------
# Module-level sentinel
# ---------------------------------------------------------------------------


def test_no_review_usable_as_module_level_sentinel():
    # Simulate: NO_REVIEW = no_review at the top of a module.
    NO_REVIEW = no_review
    assert is_no_review(NO_REVIEW) is True


# ---------------------------------------------------------------------------
# is_no_review helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj, expected",
    [
        (no_review, True),
        (_NoReview(), True),  # fresh instance also carries the class attr
        (lambda: None, False),
        (object(), False),
        (42, False),
        ("string", False),
    ],
)
def test_is_no_review_parametrized(obj, expected):
    assert is_no_review(obj) is expected


def test_is_no_review_true_for_decorated_function():
    @no_review
    def fn():
        pass

    assert is_no_review(fn) is True


def test_is_no_review_true_for_decorated_class():
    @no_review
    class Cls:
        pass

    assert is_no_review(Cls) is True


def test_is_no_review_false_for_plain_function():
    def fn():
        pass

    assert is_no_review(fn) is False


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_no_review_repr():
    assert repr(no_review) == "no_review"