from typing import List
from datetime import datetime
from util.task_model import Task, TimeGranularity, ProductType


class TaskGenerator:
    """任务生成器类，支持多种方式生成任务列表"""
    
    def __init__(self):
        self.task_id_counter = 1
    
    def generate_from_stock_list(self, 
                               stock_symbols: List[str],
                               start_prediction_timestamp: datetime = None,
                               time_granularity: TimeGranularity = TimeGranularity.DAY_1,
                               product_type: ProductType = ProductType.STOCK,
                               **kwargs) -> List[Task]:
        """根据股票列表生成任务列表
        
        Args:
            stock_symbols: 股票符号列表，如 ['BTC', 'USDT']
            start_prediction_timestamp: 开始预测时间戳，默认为当前时间
            time_granularity: 时间粒度，默认为1天
            product_type: 产品类型，默认为股票
            **kwargs: 其他任务参数
            
        Returns:
            List[Task]: 生成的任务列表
        """
        if start_prediction_timestamp is None:
            start_prediction_timestamp = datetime.now()
            
        tasks = []
        for symbol in stock_symbols:
            task = Task(
                id=self.task_id_counter,
                product_type=product_type,
                stock_symbol=symbol,
                start_prediction_timestamp=start_prediction_timestamp,
                time_granularity=time_granularity,
                **kwargs
            )
            tasks.append(task)
            self.task_id_counter += 1
            
        return tasks