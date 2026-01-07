from typing import Any, List, Optional, Tuple
from pathlib import Path
import sqlite3


class DatabaseError(Exception):
    """Custom exception for database operations."""

    pass


class DatabaseManager:
    """Simplified SQLite database manager with comprehensive error handling."""

    def __init__(self, db_path: str):
        """
        Initialize the DatabaseManager.

        Args:
            db_path: Path to the SQLite database file.

        Raises:
            DatabaseError: If database setup fails.
        """
        self.db_path = str(db_path)

    def _execute(self, query: str, values: Tuple[Any, ...]) -> sqlite3.Cursor:
        """
        Internal method to execute a query with parameters using context manager.

        Args:
            query: SQL query string.
            values: Tuple of values to bind.

        Returns:
            sqlite3.Cursor: Cursor after execution (detached from closed connection).

        Raises:
            DatabaseError: If execution fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()

                result_cursor = (
                    cursor.fetchall()
                    if query.strip().upper().startswith("SELECT")
                    else None
                )
                lastrowid = cursor.lastrowid
                rowcount = cursor.rowcount

            class CursorResult:
                def __init__(self, rows=None, lastrowid=None, rowcount=None):
                    self._rows = rows
                    self.lastrowid = lastrowid
                    self.rowcount = rowcount

                def fetchall(self):
                    return self._rows or []

            return CursorResult(result_cursor, lastrowid, rowcount)

        except sqlite3.IntegrityError as e:
            raise DatabaseError(f"Constraint violation: {e}") from e
        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()
            if "locked" in error_msg:
                raise DatabaseError("Database is locked - try again later") from e
            elif "no such table" in error_msg:
                raise DatabaseError(f"Table does not exist: {e}") from e
            elif "syntax error" in error_msg:
                raise DatabaseError(f"SQL syntax error: {e}") from e
            else:
                raise DatabaseError(f"Operational error: {e}") from e
        except sqlite3.DatabaseError as e:
            raise DatabaseError(f"Database error: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error during execution: {e}") from e

    def insert_many(self, query: str, values_list: List[Tuple[Any, ...]]) -> int:
        """
        Executes a batch INSERT using executemany.

        Args:
            query: SQL INSERT query with placeholders.
            values_list: List of tuples with values to insert.

        Returns:
            Number of rows inserted.

        Raises:
            DatabaseError: If insert fails or query is not INSERT.
        """
        if not query.strip().upper().startswith("INSERT"):
            raise DatabaseError("Query must be an INSERT statement")

        if not isinstance(values_list, list) or not all(
            isinstance(item, tuple) for item in values_list
        ):
            raise DatabaseError("Values must be a list of tuples")

        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.executemany(query, values_list)
                conn.commit()
                return cursor.rowcount
        except sqlite3.IntegrityError as e:
            raise DatabaseError(f"Constraint violation: {e}") from e
        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()
            if "locked" in error_msg:
                raise DatabaseError("Database is locked - try again later") from e
            elif "no such table" in error_msg:
                raise DatabaseError(f"Table does not exist: {e}") from e
            elif "syntax error" in error_msg:
                raise DatabaseError(f"SQL syntax error: {e}") from e
            else:
                raise DatabaseError(f"Database operation failed: {e}") from e
        except sqlite3.DatabaseError as e:
            raise DatabaseError(f"Database error: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error during batch insert: {e}") from e

    def insert(self, query: str, values: Tuple[Any, ...]) -> int:
        """
        Execute an INSERT statement.
        Args:
            query: SQL INSERT query.
            values: Values to insert.
        Returns:
            Row ID of the inserted record.
        Raises:
            DatabaseError: If insert fails or query is not INSERT.
        """
        if not query.strip().upper().startswith("INSERT"):
            raise DatabaseError("Query must be an INSERT statement")

        cursor = self._execute(query, values)
        row_id = cursor.lastrowid

        if row_id is None:
            raise DatabaseError("INSERT did not return a row ID")

        return row_id

    def select(
        self, query: str, values: Optional[Tuple[Any, ...]] = None
    ) -> List[sqlite3.Row]:
        """
        Execute a SELECT statement.

        Args:
            query: SQL SELECT query.
            values: Query parameters.

        Returns:
            List of result rows.

        Raises:
            DatabaseError: If select fails or query is not SELECT.
        """
        if not query.strip().upper().startswith("SELECT"):
            raise DatabaseError("Query must be a SELECT statement")

        cursor = self._execute(query, values or ())
        return cursor.fetchall()

    def update(self, query: str, values: Tuple[Any, ...]) -> int:
        """
        Execute an UPDATE statement.

        Args:
            query: SQL UPDATE query.
            values: Values for the update.

        Returns:
            Number of affected rows.

        Raises:
            DatabaseError: If update fails or query is not UPDATE.
        """
        if not query.strip().upper().startswith("UPDATE"):
            raise DatabaseError("Query must be an UPDATE statement")

        cursor = self._execute(query, values)
        return cursor.rowcount

    def delete(self, query: str, values: Tuple[Any, ...]) -> int:
        """
        Execute a DELETE statement.

        Args:
            query: SQL DELETE query.
            values: Values to identify records to delete.

        Returns:
            Number of deleted rows.

        Raises:
            DatabaseError: If delete fails or query is not DELETE.
        """
        if not query.strip().upper().startswith("DELETE"):
            raise DatabaseError("Query must be a DELETE statement")

        cursor = self._execute(query, values)
        return cursor.rowcount

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.

        Raises:
            DatabaseError: If table name is invalid.
        """
        if not table_name or not table_name.strip():
            raise DatabaseError("Table name cannot be empty")

        if not table_name.replace("_", "").replace("-", "").isalnum():
            raise DatabaseError("Table name contains invalid characters")

        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.select(query, (table_name.strip(),))
        return len(result) > 0

    def get_table_info(self, table_name: str) -> List[sqlite3.Row]:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column information.

        Raises:
            DatabaseError: If table doesn't exist or name is invalid.
        """
        if not self.table_exists(table_name):
            raise DatabaseError(f"Table '{table_name}' does not exist")

        query = f"PRAGMA table_info({table_name})"
        return self.select(query)

    def close(self) -> None:
        """
        Explicitly close database connections.
        Note: Not needed with context managers, but provided for completeness.
        """
        self.connection.close()