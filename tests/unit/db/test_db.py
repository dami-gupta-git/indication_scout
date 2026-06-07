"""Unit tests for the session layer."""

from unittest.mock import MagicMock, patch


def test_get_db_yields_and_closes():
    """get_db should yield a session and close it on exit."""
    mock_session = MagicMock()
    mock_factory = MagicMock(return_value=mock_session)

    with patch(
        "indication_scout.db.session._make_session_factory",
        return_value=mock_factory,
    ):
        from indication_scout.db.session import get_db

        gen = get_db()
        session = next(gen)
        assert session is mock_session

        try:
            next(gen)
        except StopIteration:
            pass

        mock_session.close.assert_called_once()
