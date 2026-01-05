from pathlib import Path
from yfinance import Ticker
from pandas import DataFrame
from os import path, makedirs

from logging import getLogger, basicConfig, INFO    


basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

ROOT_DIR = path.abspath(path.join(path.dirname(__file__), '..', '..'))
logger.info(f"ROOT_DIR: {ROOT_DIR}")

def extract_nvidia_data(period: str ='max', interval: str ='1d') -> DataFrame:
    """
    Extracts historical NVIDIA stock data using yfinance.

    Args:
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo')

    Note: For long periods (like 'max'), use interval '1d' or higher.
          Intraday data (1m, 5m, etc.) is limited to the last 7-60 days.

    Returns:
        DataFrame with historical NVIDIA data
    """
    ticker: str = 'NVDA'
    logger.info(f"Connecting to Yahoo Finance for {ticker}...")

    nvidia: Ticker = Ticker(ticker)

    logger.info(f"Downloading historical data (period: {period}, interval: {interval})...")
    data: DataFrame = nvidia.history(period=period, interval=interval)
    if data.empty:
        logger.warning("No data was returned. Check the parameters.")
        return None

    data.reset_index(inplace=True)

    if 'Datetime' in data.columns:
        data.rename(columns={'Datetime': 'Date'}, inplace=True)

    return data


def save_data(data: DataFrame, path: str ='nvidia_stock.csv') -> None:
    """
    Saves the data to a CSV file.

    Args:
        data: DataFrame with the data
        path: Path to save the file
    """
    full_path = Path(path) if path.startswith('/') else Path(ROOT_DIR, "data", "raw", path)
    dir_name = full_path.parent
    if not dir_name.exists():
        dir_name.mkdir(parents=True, exist_ok=True)
    data.to_csv(full_path, index=False)
    logger.info(f"Data saved to {full_path}")

def show_statistics(data: DataFrame) -> None:
    logger.info("="*60)
    logger.info("DATA STATISTICS")
    logger.info("="*60)
    logger.info(f"Total records: {len(data)}")
    logger.info(f"Period: {data['Date'].min()} to {data['Date'].max()}")
    logger.info("\nFirst 5 rows:\n%s", data.head().to_string())
    logger.info("\nLast 5 rows:\n%s", data.tail().to_string())
    logger.info("\nStatistical summary:\n%s", data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("NVIDIA (NVDA) HISTORICAL DATA EXTRACTOR")
    logger.info("="*60)

    df = extract_nvidia_data(period='max', interval='1d')

    if df is not None:
        logger.info("Data extraction completed successfully!")

        show_statistics(df)
        logger.info("="*60)
        save_data(df, 'nvidia_stock_max.csv')
        logger.info("="*60)
    else:
        logger.error("Data extraction failed.")

    logger.info("Process finished!")