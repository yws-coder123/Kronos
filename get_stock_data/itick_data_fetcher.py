"""iTick数据获取器，用于根据任务参数获取股票数据"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import time
from dotenv import load_dotenv

# 导入iTick API
from util.itick_api import get_kline_data
from util.task_model import ProductType, TimeGranularity

# 加载环境变量
load_dotenv()


class ITickDataFetcher:
    """iTick数据获取器类"""
    
    def __init__(self):
        """
        初始化数据获取器
        """
        # 检查API密钥
        api_key = os.getenv('ITICK_API_KEY')
        if not api_key:
            raise ValueError("未找到ITICK_API_KEY，请设置环境变量")
        
        # 速率限制：根据iTick API限制调整
        self.rate_limit_delay = 1  # 调用间隔秒数
        self.last_call_time = 0
        
        # 时间粒度映射到iTick k_type
        self.time_granularity_map = {
            TimeGranularity.MINUTE_1: 1,   # 1分钟
            TimeGranularity.MINUTE_5: 2,   # 5分钟
            TimeGranularity.MINUTE_15: 3,  # 15分钟
            TimeGranularity.MINUTE_30: 4,  # 30分钟
            TimeGranularity.HOUR_1: 5,     # 1小时
            TimeGranularity.DAY_1: 8,      # 1天
            TimeGranularity.WEEK_1: 9,     # 1周
            TimeGranularity.MONTH_1: 10,   # 1月
        }
        
        # 产品类型映射
        self.product_type_map = {
            ProductType.CRYPTO: "加密货币",
            ProductType.FOREX: "外汇",
            ProductType.INDICES: "指数",
            ProductType.STOCK: "股票",
            ProductType.FUTURE: "期货",
            ProductType.FUND: "基金"
        }
    
    def _rate_limit_check(self):
        """检查速率限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            print(f"速率限制：等待 {sleep_time:.1f} 秒")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _get_k_type(self, time_granularity: TimeGranularity) -> int:
        """
        获取iTick API的k_type
        
        Args:
            time_granularity: 时间粒度枚举
            
        Returns:
            k_type: iTick API的时间类型
        """
        if time_granularity not in self.time_granularity_map:
            raise ValueError(f"不支持的时间粒度: {time_granularity}")
        
        return self.time_granularity_map[time_granularity]
    
    def _get_product_type_str(self, product_type: ProductType) -> str:
        """
        获取iTick API的产品类型字符串
        
        Args:
            product_type: 产品类型枚举
            
        Returns:
            product_type_str: iTick API的产品类型字符串
        """
        if product_type not in self.product_type_map:
            raise ValueError(f"不支持的产品类型: {product_type}")
        
        return self.product_type_map[product_type]
    
    def _parse_symbol(self, symbol: str, product_type: ProductType) -> Tuple[str, str]:
        """
        解析股票代码，提取region和code
        
        Args:
            symbol: 股票代码
            product_type: 产品类型
            
        Returns:
            (region, code): 市场代码和产品代码
        """
        # 根据产品类型和符号格式解析
        if product_type == ProductType.CRYPTO:
            # 加密货币通常使用BA市场
            return "BA", symbol
        elif product_type == ProductType.FOREX:
            # 外汇通常使用BA市场
            return "BA", symbol
        elif product_type == ProductType.STOCK:
            # 股票可能需要根据交易所解析，这里简化处理
            return "US", symbol  # 假设美股
        else:
            # 其他类型默认使用BA
            return "BA", symbol
    
    def calculate_end_timestamp_and_limit(self, task) -> Tuple[int, int]:
        """
        根据任务参数计算结束时间戳和数据条数
        
        Args:
            task: 任务对象，包含start_prediction_timestamp, input_length, output_length
            
        Returns:
            (end_timestamp_ms, total_limit): 结束时间戳（毫秒）和总数据条数
        """
        # iTick API需要结束时间戳（毫秒）和数据条数
        # 我们需要获取 input_length + output_length 条数据
        # 结束时间戳应该是 start_prediction_timestamp + output_length * time_granularity
        # 这样iTick会返回包含未来数据的完整数据集
        
        # 解析时间粒度
        time_granularity_str = task.time_granularity
        
        # 计算时间偏移
        import re
        match = re.match(r'(\d+)([mhdwM])', time_granularity_str)
        if not match:
            raise ValueError(f"无法解析时间粒度: {time_granularity_str}")
        
        value = int(match.group(1))
        unit = match.group(2)
        
        # 根据单位创建timedelta对象
        if unit == 'm':
            end_offset = timedelta(minutes=value * task.output_length)
        elif unit == 'h':
            end_offset = timedelta(hours=value * task.output_length)
        elif unit == 'd':
            end_offset = timedelta(days=value * task.output_length)
        elif unit == 'w':
            end_offset = timedelta(weeks=value * task.output_length)
        elif unit == 'M':
            # 月份处理比较复杂，这里简化为30天
            end_offset = timedelta(days=30 * value * task.output_length)
        else:
            raise ValueError(f"不支持的时间单位: {unit}")
        
        # 计算结束时间戳
        end_timestamp = task.start_prediction_timestamp + end_offset
        end_timestamp_ms = int(end_timestamp.timestamp() * 1000)
        
        # 总数据条数
        total_limit = task.input_length + task.output_length
        
        print(f"计算时间戳: start_prediction_timestamp={task.start_prediction_timestamp}, end_timestamp={end_timestamp}, end_timestamp_ms={end_timestamp_ms}")
        
        return end_timestamp_ms, total_limit
    
    def fetch_and_split_stock_data(self, task) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        根据任务参数获取股票数据并按时间戳分割
        
        Args:
            task: 任务对象
            
        Returns:
            (historical_data, future_data): 历史数据和未来数据的DataFrame元组
        """
        try:
            # 计算结束时间戳和数据条数
            end_timestamp_ms, total_limit = self.calculate_end_timestamp_and_limit(task)
            
            # 获取k_type
            k_type = self._get_k_type(task.time_granularity)
            
            # 获取产品类型字符串
            product_type_str = self._get_product_type_str(task.product_type)
            
            # 解析股票代码
            region, code = self._parse_symbol(task.stock_symbol, task.product_type)
            
            print(f"获取股票数据: {task.stock_symbol}, 产品类型: {product_type_str}, 结束时间戳: {end_timestamp_ms}, 数据条数: {total_limit}, 粒度: {task.time_granularity}")
            
            # 速率限制检查
            self._rate_limit_check()
            
            # 调用iTick API获取数据
            response = get_kline_data(
                product_type=product_type_str,
                region=region,
                code=code,
                k_type=k_type,
                limit=total_limit,
                et=str(end_timestamp_ms)
            )
            
            if response.get('code') != 0 or not response.get('data'):
                print(f"警告：未获取到数据: {task.stock_symbol}, 响应码: {response.get('code')}, 消息: {response.get('msg', '未知错误')}")
                return None, None
            
            # 转换为DataFrame
            data_list = []
            for item in response['data']:
                data_list.append({
                    'timestamp': datetime.fromtimestamp(item['t'] / 1000),
                    'open': item['o'],
                    'high': item['h'],
                    'low': item['l'],
                    'close': item['c'],
                    'volume': item['v'],
                    'turnover': item.get('tu', 0),  # 成交金额
                })
            
            df = pd.DataFrame(data_list)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 根据start_prediction_timestamp分割数据
            historical_data = df[df['timestamp'] <= task.start_prediction_timestamp].copy()
            future_data = df[df['timestamp'] > task.start_prediction_timestamp].copy()
            
            print(f"成功获取 {len(df)} 条数据记录")
            print(f"历史数据: {len(historical_data)} 条, 未来数据: {len(future_data)} 条")
            
            if len(df) > 0:
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
                'vwap': df.get('vwap', df['close']),  # 如果没有vwap，使用close价格
                'transactions': df.get('transactions', 0),  # 如果没有transactions，使用0
                'turnover': df.get('turnover', 0)  # 成交金额
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