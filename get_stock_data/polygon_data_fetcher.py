"""Polygon.io数据获取器，用于根据任务参数获取股票数据"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from polygon import RESTClient
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class PolygonDataFetcher:
    """Polygon.io数据获取器类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化数据获取器
        
        Args:
            api_key: Polygon.io API密钥，如果不提供则从环境变量获取
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("未找到POLYGON_API_KEY，请设置环境变量或传入api_key参数")
        
        self.client = RESTClient(self.api_key)
        
        # 速率限制：免费版每分钟5次调用
        self.rate_limit_delay = 12  # 调用间隔秒数
        self.last_call_time = 0
    
    def _rate_limit_check(self):
        """检查速率限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            print(f"速率限制：等待 {sleep_time:.1f} 秒")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _parse_time_granularity(self, time_granularity: str) -> tuple:
        """
        解析时间粒度字符串
        
        Args:
            time_granularity: 时间粒度字符串，如 '5m', '1h', '1d'
            
        Returns:
            (multiplier, timespan, timedelta_kwargs): 倍数、时间单位、timedelta参数字典
        """
        granularity_map = {
            '1m': (1, 'minute', {'minutes': 1}),
            '5m': (5, 'minute', {'minutes': 5}),
            '15m': (15, 'minute', {'minutes': 15}),
            '30m': (30, 'minute', {'minutes': 30}),
            '1h': (1, 'hour', {'hours': 1}),
            '2h': (2, 'hour', {'hours': 2}),
            '4h': (4, 'hour', {'hours': 4}),
            '6h': (6, 'hour', {'hours': 6}),
            '12h': (12, 'hour', {'hours': 12}),
            '1d': (1, 'day', {'days': 1}),
            '3d': (3, 'day', {'days': 3}),
            '1w': (1, 'week', {'weeks': 1}),
            '2w': (2, 'week', {'weeks': 2}),
        }
        
        if time_granularity not in granularity_map:
            raise ValueError(f"不支持的时间粒度: {time_granularity}")
        
        return granularity_map[time_granularity]
    
    def calculate_timestamps(self, task) -> tuple:
        """
        根据任务参数计算开始和结束时间戳
        
        Args:
            task: 任务对象，包含start_prediction_timestamp, time_granularity, input_length, output_length
            
        Returns:
            (start_timestamp, end_timestamp): 开始和结束时间戳
        """
        # 解析时间粒度，提取数值和单位
        import re
        match = re.match(r'(\d+)([mhdw])', task.time_granularity)
        if not match:
            raise ValueError(f"无法解析时间粒度: {task.time_granularity}")
        
        value = int(match.group(1))
        unit = match.group(2)
        
        # 根据单位创建timedelta对象
        if unit == 'm':
            # 分钟
            start_offset = timedelta(minutes=value * task.input_length)
            end_offset = timedelta(minutes=value * task.output_length)
        elif unit == 'h':
            # 小时
            start_offset = timedelta(hours=value * task.input_length)
            end_offset = timedelta(hours=value * task.output_length)
        elif unit == 'd':
            # 天
            start_offset = timedelta(days=value * task.input_length)
            end_offset = timedelta(days=value * task.output_length)
        elif unit == 'w':
            # 周
            start_offset = timedelta(weeks=value * task.input_length)
            end_offset = timedelta(weeks=value * task.output_length)
        else:
            raise ValueError(f"不支持的时间单位: {unit}")
        
        # 计算开始和结束时间戳
        start_timestamp = task.start_prediction_timestamp - start_offset
        end_timestamp = task.start_prediction_timestamp + end_offset
        
        return start_timestamp, end_timestamp
    
    def fetch_and_split_stock_data(self, task) -> tuple:
        """
        根据任务参数获取股票数据并按时间戳分割
        
        Args:
            task: 任务对象
            
        Returns:
            (historical_data, future_data): 历史数据和未来数据的DataFrame元组
        """
        try:
            # 计算时间戳
            start_timestamp, end_timestamp = self.calculate_timestamps(task)
            
            # 解析时间粒度
            multiplier, timespan, _ = self._parse_time_granularity(task.time_granularity)
            
            print(f"获取股票数据: {task.stock_symbol}, 从 {start_timestamp} 到 {end_timestamp}, 粒度: {task.time_granularity}")
            
            # 速率限制检查
            self._rate_limit_check()
            
            # 调用Polygon API获取数据
            aggs = list(self.client.get_aggs(
                ticker=task.stock_symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_timestamp,
                to=end_timestamp,
                limit=5000,


                adjusted=True,
                sort="asc",
            ))
            
            if not aggs:
                print(f"警告：未获取到数据: {task.stock_symbol}")
                return None
            
            # 转换为DataFrame
            data_list = []
            for agg in aggs:
                data_list.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': getattr(agg, 'vwap', None),
                    'transactions': getattr(agg, 'transactions', None)
                })
            
            df = pd.DataFrame(data_list)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
  
            
            # 根据start_prediction_timestamp分割数据
            historical_data = df[df['timestamp'] < task.start_prediction_timestamp].copy()
            future_data = df[df['timestamp'] >= task.start_prediction_timestamp].copy()
            
            print(f"成功获取 {len(df)} 条数据记录")
            print(f"历史数据: {len(historical_data)} 条, 未来数据: {len(future_data)} 条")
            min_timestamp = df['timestamp'].min()
            max_timestamp = df['timestamp'].max()
            print(f"获取到数据的时间戳范围: 最小 {min_timestamp}, 最大 {max_timestamp}")
            
            return historical_data, future_data
            
        except Exception as e:
            print(f"错误：获取股票数据失败: {str(e)}")
            return None, None
    
    def fetch_and_process_data(self, task):
        """
        获取数据并处理后保存到task对象中
        
        Args:
            task: 任务对象
        """
        historical_data, future_data = self.fetch_and_split_stock_data(task)
        
        if historical_data is not None:
            task.historical_stock_data = self.format_for_kronos(historical_data, task.stock_symbol)
        
        if future_data is not None:
            task.future_stock_data = self.format_for_kronos(future_data, task.stock_symbol)
        
        return task
    
    def format_for_kronos(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        将数据格式化为Kronos模型所需格式
        
        Args:
            df: 原始数据DataFrame
            symbol: 股票代码
            
        Returns:
            格式化后的DataFrame
        """
        if df is None or df.empty:
            return None
        
        try:
            # 创建Kronos格式的DataFrame
            kronos_df = pd.DataFrame({
                'symbol': symbol,
                'timestamp': df['timestamp'],
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume'],
                'vwap': df['vwap'],
                'transactions': df['transactions']
            })
            
            # 计算技术指标
            kronos_df['price_change'] = kronos_df['close'].pct_change()
            kronos_df['high_low_ratio'] = kronos_df['high'] / kronos_df['low']
            kronos_df['volume_ma'] = kronos_df['volume'].rolling(window=5, min_periods=1).mean()
            
            # 填充NaN值
            kronos_df = kronos_df.ffill().fillna(0)
            
            return kronos_df
            
        except Exception as e:
            print(f"错误：格式化数据失败: {str(e)}")
            return None