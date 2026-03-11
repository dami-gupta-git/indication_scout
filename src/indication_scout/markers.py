"""Code review markers.

Provides the ``no_review`` marker, which signals to the code review agent that
the decorated or annotated item should be excluded from review.

Usage
-----
Function / method / class::

    from indication_scout.markers import no_review

    @no_review
    def legacy_helper(...):
        ...

    @no_review
    class LegacyModel:
        ...

Pydantic / dataclass field (via ``typing.Annotated``)::

    from typing import Annotated
    from indication_scout.markers import no_review

    class MyModel(BaseModel):
        internal_id: Annotated[str, no_review] = ""

Module-level exclusion (place near the top of the file)::

    from indication_scout.markers import no_review
    NO_REVIEW = no_review

The code review agent detects exclusions by:

* Checking ``getattr(obj, '__no_review__', False)`` on callables and classes.
* Scanning ``typing.get_type_hints(..., include_extras=True)`` for
  ``Annotated`` metadata containing ``no_review``.
* Detecting a module-level name bound to ``no_review`` via AST or
  ``vars(module)`` inspection.
"""

from __future__ import annotations

from typing import Any, TypeVar  # noqa: F401 – Any used in is_no_review, TypeVar in _T

_T = TypeVar("_T")


class _NoReview:
    """Singleton sentinel used as the ``no_review`` marker.

    Acts as a no-op decorator for callables and classes (sets
    ``__no_review__ = True``), and as an opaque object suitable for use as
    ``typing.Annotated`` metadata on fields.
    """

    __slots__ = ()

    def __call__(self, obj: _T) -> _T:
        """Decorate a callable or class; attach the marker attribute."""
        obj.__no_review__ = True  # type: ignore[attr-defined]
        return obj

    def __repr__(self) -> str:
        return "no_review"

    # Make the singleton itself carry the marker so module-level detection
    # (``NO_REVIEW = no_review``) works without extra logic.
    __no_review__: bool = True


no_review: _NoReview = _NoReview()


def is_no_review(obj: Any) -> bool:
    """Return True if *obj* carries the no_review marker.

    Works for:
    * Decorated callables / classes (checks ``__no_review__`` attribute).
    * The ``no_review`` singleton itself (used as ``Annotated`` metadata or
      as a module-level sentinel).
    """
    return bool(getattr(obj, "__no_review__", False))
