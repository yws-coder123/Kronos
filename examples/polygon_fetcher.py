"""
Polygon.io Data Fetcher Utility for Kronos Financial Model

This module provides a utility class to fetch financial market data from Polygon.io
in the format required by the Kronos prediction model using the official Polygon Python client.
"""

from polygon import RESTClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonDataFetcher:
    """
    Utility class to fetch S&P 500 stock data from Polygon.io API
    and format it for use with Kronos prediction models.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the PolygonDataFetcher using official Polygon Python client.
        
        Args:
            api_key (str): Your Polygon.io API key
        """
        self.api_key = api_key
        self.client = RESTClient(api_key)
        
        # Rate limiting: Free tier allows 5 calls per minute
        self.rate_limit_delay = 12  # seconds between calls for free tier
        self.last_call_time = 0
        
        # S&P 500 symbols (sample of major companies)
        self.sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
            'VZ', 'ADBE', 'CRM', 'CMCSA', 'PFE', 'NKE', 'T', 'INTC', 'WMT',
            'KO', 'CSCO', 'XOM', 'ABT', 'CVX', 'TMO', 'ACN', 'COST', 'AVGO',
            'DHR', 'NEE', 'LLY', 'TXN', 'QCOM', 'HON', 'LOW', 'UPS', 'BMY',
            'PM', 'SBUX', 'LMT', 'IBM'
        ]
    
    def _rate_limit_check(self):
        """Ensure we don't exceed API rate limits."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def get_sp500_symbols(self) -> List[str]:
        """
        Get list of available S&P 500 symbols.
        
        Returns:
            List[str]: List of stock symbols
        """
        return self.sp500_symbols.copy()
    
    def fetch_aggregates(self, 
                        symbol: str, 
                        multiplier: int = 5,
                        timespan: str = "minute",
                        from_date: str = None,
                        to_date: str = None,
                        limit: int = 5000) -> Optional[List[Any]]:
        """
        Fetch aggregate data using Polygon.io official Python client.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            multiplier (int): Size of the timespan multiplier
            timespan (str): Size of the time window ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format  
            limit (int): Number of results to return (max 50000)
            
        Returns:
            Optional[List]: List of aggregate data or None if failed
        """
        self._rate_limit_check()
        
        # Default date range: last 30 days
        if not from_date or not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        try:
            logger.info(f"Fetching data for {symbol} from {from_date} to {to_date} using Polygon Python client")
            
            # Use the official Polygon client to get aggregates
            aggregates = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                adjusted=True,
                sort="asc",
                limit=limit
            )
            
            # Convert generator to list
            agg_list = list(aggregates)
            
            if agg_list:
                logger.info(f"Successfully fetched {len(agg_list)} records for {symbol}")
                return agg_list
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def convert_to_kronos_format(self, 
                                agg_list: List[Any], 
                                symbol: str,
                                multiplier: int = 5) -> Optional[pd.DataFrame]:
        """
        Convert Polygon.io aggregate objects to Kronos model input format.
        
        Args:
            agg_list (List): List of aggregate objects from Polygon client
            symbol (str): Stock symbol
            multiplier (int): Timespan multiplier used in the API call
            
        Returns:
            Optional[pd.DataFrame]: Formatted DataFrame or None if conversion failed
        """
        try:
            if not agg_list:
                logger.warning(f"No aggregate data found for {symbol}")
                return None
            
            # Convert to DataFrame
            df_data = []
            for agg in agg_list:
                # Polygon aggregate object has attributes: timestamp, open, high, low, close, volume, etc.
                # Timestamp is already in seconds (not milliseconds like raw API)
                timestamp = datetime.fromtimestamp(agg.timestamp / 1000)  # Convert from milliseconds
                
                row = {
                    'timestamps': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(agg.open),
                    'high': float(agg.high),
                    'low': float(agg.low),
                    'close': float(agg.close),
                    'volume': float(agg.volume),
                    'amount': float(agg.volume) * float(agg.close)  # Volume * Close as proxy for amount
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Ensure data is sorted by timestamp
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            logger.info(f"Successfully converted {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error converting data for {symbol}: {str(e)}")
            return None
    
    def get_stock_data(self, 
                      symbol: str, 
                      current_timestamp: str = None,
                      lookback_days: int = 30,
                      multiplier: int = 5,
                      timespan: str = "minute") -> Optional[pd.DataFrame]:
        """
        Main method to fetch stock data in Kronos format.
        
        Args:
            symbol (str): Stock symbol (must be in S&P 500)
            current_timestamp (str): Current timestamp in 'YYYY-MM-DD HH:MM:SS' format
            lookback_days (int): Number of days to look back from current timestamp
            multiplier (int): Timespan multiplier (default 5 for 5-minute intervals)
            timespan (str): Timespan type ('minute', 'hour', 'day')
            
        Returns:
            Optional[pd.DataFrame]: Formatted data ready for Kronos model or None if failed
        """
        # Validate symbol
        if symbol not in self.sp500_symbols:
            logger.error(f"Symbol {symbol} not found in S&P 500 list")
            return None
        
        # Set date range
        if current_timestamp:
            try:
                current_dt = datetime.strptime(current_timestamp, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                logger.error(f"Invalid timestamp format: {current_timestamp}. Expected: YYYY-MM-DD HH:MM:SS")
                return None
        else:
            current_dt = datetime.now()
        
        to_date = current_dt.strftime('%Y-%m-%d')
        from_date = (current_dt - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # Fetch data from Polygon.io
        agg_data = self.fetch_aggregates(
            symbol=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date
        )
        
        # If minute data fails, try daily data as fallback
        if not agg_data and timespan == "minute":
            logger.info(f"Minute data not available for {symbol}, trying daily data as fallback")
            agg_data = self.fetch_aggregates(
                symbol=symbol,
                multiplier=1,
                timespan="day",
                from_date=from_date,
                to_date=to_date
            )
            if agg_data:
                logger.info("Successfully fetched daily data as fallback")
        
        if not agg_data:
            return None
        
        # Convert to Kronos format
        df = self.convert_to_kronos_format(agg_data, symbol, multiplier)
        
        if df is not None and len(df) > 0:
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            logger.info(f"Date range: {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str, output_dir: str = "data") -> str:
        """
        Save the fetched data to CSV file.
        
        Args:
            df (pd.DataFrame): Data to save
            symbol (str): Stock symbol
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to: {filepath}")
        
        return filepath


def main():
    """
    Example usage of PolygonDataFetcher
    """
    # Example usage - you need to set your Polygon.io API key
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key:
        print("Please set POLYGON_API_KEY environment variable")
        return
    
    fetcher = PolygonDataFetcher(api_key)
    
    # Get Apple stock data
    df = fetcher.get_stock_data('AAPL', lookback_days=7)
    
    if df is not None:
        print("Sample data:")
        print(df.head())
        print(f"\nTotal records: {len(df)}")
        
        # Save to file
        filepath = fetcher.save_data(df, 'AAPL')
        print(f"Data saved to: {filepath}")
    else:
        print("Failed to fetch data")


if __name__ == "__main__":
    main()