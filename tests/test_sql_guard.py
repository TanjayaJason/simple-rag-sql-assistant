from unittest.mock import MagicMock

import pytest

from app.sql_guard import execute_read_query, validate_read_query


@pytest.mark.parametrize("sql", [
    "DELETE FROM students",
    "SELECT 1; DELETE FROM students",
    "SELECT 1; SELECT 2",
    "",
])
def test_rejects_non_read_or_multiple_statements(sql):
    with pytest.raises(ValueError):
        validate_read_query(sql)


def test_allows_single_select_and_cte():
    assert validate_read_query("SELECT 1;") == "SELECT 1;"
    assert validate_read_query("WITH sample AS (SELECT 1) SELECT * FROM sample").startswith("WITH")


def test_query_runs_in_read_only_transaction(monkeypatch):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [{"count": 8}]
    monkeypatch.setattr("app.sql_guard.psycopg2.connect", lambda **kwargs: connection)

    assert execute_read_query("SELECT count(*) FROM students") == [{"count": 8}]
    connection.set_session.assert_called_once_with(readonly=True)
    assert cursor.execute.call_args_list[0].args == ("SET LOCAL statement_timeout = 10000",)
    assert cursor.execute.call_args_list[1].args == ("SELECT count(*) FROM students",)
    connection.close.assert_called_once()
