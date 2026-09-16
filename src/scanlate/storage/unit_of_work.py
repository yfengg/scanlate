"""Transaction ownership sits here, above the repositories.

Repository methods issue statements; they do not commit. A domain action that
touches several tables — approval writes segment state, memory, an occurrence
row and possibly a glossary decision — wraps them in one transaction so a
failure anywhere rolls back all of it.

Connections are in autocommit mode, so a single write outside a transaction
still persists; the boundary is only needed where atomicity matters.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator


class UnitOfWork:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._depth = 0

    @property
    def active(self) -> bool:
        return self._depth > 0

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Reentrant: nested blocks join the outermost transaction."""
        outermost = self._depth == 0
        if outermost:
            self.conn.execute("BEGIN")
        self._depth += 1
        try:
            yield self.conn
        except Exception:
            self._depth -= 1
            if self._depth == 0 and self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0 and self.conn.in_transaction:
                self.conn.execute("COMMIT")
