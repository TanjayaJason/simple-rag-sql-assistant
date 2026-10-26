"""Validate and run generated SQL with database enforced read-only access."""

import os
from contextlib import closing

import psycopg2
import sqlparse
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from sqlparse.tokens import DML, Keyword

load_dotenv()


def validate_read_query(sql: str) -> str:
    statements = sqlparse.parse(sql)
    if len(statements) != 1:
        raise ValueError("Generated SQL must be a single SELECT query")
    statement = statements[0]
    tokens = [token for token in statement.tokens if not token.is_whitespace]
    is_select = statement.get_type() == "SELECT"
    is_cte_select = (
        tokens and tokens[0].ttype in Keyword and tokens[0].normalized == "WITH"
        and any(token.ttype in DML and token.normalized == "SELECT" for token in tokens[1:])
    )
    if not (is_select or is_cte_select):
        raise ValueError("Generated SQL must be a single SELECT query")
    return sql.strip()


def execute_read_query(sql: str) -> list[dict]:
    sql = validate_read_query(sql)
    # A separate, SELECT-only PostgreSQL role should be configured in production.
    with closing(psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_READ_USER") or os.getenv("DB_USER"),
        password=os.getenv("DB_READ_PASSWORD") or os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )) as conn:
        conn.set_session(readonly=True)
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SET LOCAL statement_timeout = 10000")
                cur.execute(sql)
                return [dict(row) for row in cur.fetchall()]
