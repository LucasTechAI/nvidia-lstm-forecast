from logging import getLogger, basicConfig, INFO
from sqlite3 import Connection, Error, connect
from pandas import read_csv
from os import path


CSV_PATH = path.abspath(
    path.join(path.dirname(__file__), "../../data/raw/nvidia_stock_max.csv")
)
DB_PATH = path.abspath(path.join(path.dirname(__file__), "../../data/nvidia_stock.db"))
TABLE_NAME = "nvidia_stock"

basicConfig(level=INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)


def create_table_if_not_exists(conn: Connection) -> None:
    """
    Creates the NVIDIA stock table if it does not exist.
    Args:
        conn: SQLite database connection
    """
    logger.info("Creating table if not exists")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        logger.info(f"Dropped existing table {TABLE_NAME}")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                dividends REAL,
                stock_splits REAL
            )
        """
        )
        logger.info(f"Table {TABLE_NAME} is ready")
    except Error as e:
        logger.error(f"An error occurred while creating the table: {e}")
        raise


def load_csv_to_sqlite(csv_path: str = CSV_PATH, db_path: str = DB_PATH) -> None:
    """
    Loads data from a CSV file into a SQLite database.
    Args:
        csv_path: Path to the CSV file
        db_path: Path to the SQLite database
    """
    logger.info(f"Loading data from {csv_path} to {db_path}")
    try:
        df = read_csv(csv_path)
        df.rename(columns={"Stock Splits": "stock_splits"}, inplace=True)
        df.rename(columns=lambda x: x.lower(), inplace=True)
        conn = connect(db_path)
        create_table_if_not_exists(conn)
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        if "conn" in locals():
            conn.close()
