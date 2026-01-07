from logging import getLogger, basicConfig, INFO
import os, sys

basicConfig(level=INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.etl.extrator_nvidia import extract_nvidia_data, save_data, show_statistics
from src.etl.load_sqlite_nvidia import load_csv_to_sqlite

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NVIDIA (NVDA) HISTORICAL DATA EXTRACTOR")
    logger.info("=" * 60)

    df = extract_nvidia_data(period="max", interval="1d")

    if df is not None:
        logger.info("Data extraction completed successfully!")

        show_statistics(df)
        logger.info("=" * 60)
        save_data(df, "nvidia_stock.csv")
        logger.info("=" * 60)
    else:
        logger.error("Data extraction failed.")

    logger.info("Starting data load into SQLite database...")
    try:
        load_csv_to_sqlite()
        logger.info("Data load completed successfully!")
    except Exception as e:
        logger.error(f"Data load failed: {e}")

    logger.info("Process finished!")
