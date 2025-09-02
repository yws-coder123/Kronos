"""Kronos预测模块

根据task启动预测任务，封装模型运行逻辑。
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional
import sys

# Configure matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from util.task_model import Task, TaskStatus


class KronosTaskPredictor:
    """Kronos任务预测器
    
    封装模型加载和预测逻辑，支持GPU/CPU自动检测。
    """
    
    def __init__(self, enable_plotting: bool = True):
        """
        初始化预测器
        
        Args:
            enable_plotting: 是否启用生图功能，默认开启
        """
        self.enable_plotting = enable_plotting
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        self.predictor = None
        
        print(f"使用设备: {self.device}")
    
    def _detect_device(self) -> str:
        """检测GPU是否可用，设置模型使用CPU还是GPU"""
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            print("GPU不可用，使用CPU")
            return "cpu"
    
    def _load_model(self, model_name: str):
        """加载模型和tokenizer"""
        if self.model is None or self.tokenizer is None:
            print(f"正在加载模型: {model_name}")
            try:
                self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
                self.model = Kronos.from_pretrained(f"NeoQuasar/{model_name}")
                self.predictor = KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=512)
                print("模型加载成功")
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                raise
    
    def _plot_prediction(self, task: Task, pred_df: pd.DataFrame):
        """生成预测结果图表"""
        if not self.enable_plotting:
            return
            
        try:
            # 准备数据
            historical_df = task.historical_stock_data
            future_df = task.future_stock_data
            if historical_df is None or len(historical_df) == 0:
                print("没有历史数据，跳过生图")
                return
            
            # 合并历史数据和真实未来数据 - 参考 prediction_example.py
            if future_df is not None and len(future_df) > 0:
                # 创建完整的真实数据（历史+未来）
                kline_df = pd.concat([historical_df, future_df], ignore_index=True)
                # 设置预测数据的索引，使其对应未来数据的位置
                pred_df.index = kline_df.index[-pred_df.shape[0]:]
            else:
                # 如果没有真实未来数据，只使用历史数据
                kline_df = historical_df
                pred_df.index = kline_df.index[-pred_df.shape[0]:]
            
            # 准备时间戳数据 - 严格使用数据中的时间戳
            hist_timestamps = pd.to_datetime(historical_df['timestamp'])
            if future_df is not None and len(future_df) > 0:
                future_timestamps = pd.to_datetime(future_df['timestamp'])
                all_timestamps = pd.concat([hist_timestamps, future_timestamps], ignore_index=True)
                # 预测数据的时间戳
                pred_timestamps = future_timestamps
            else:
                all_timestamps = hist_timestamps
                # 如果没有未来数据，使用历史数据的最后部分作为预测时间戳
                pred_timestamps = hist_timestamps.iloc[-len(pred_df):]
            
            # 准备绘图数据 - 使用实际时间戳
            sr_close = pd.Series(kline_df['close'].values, index=all_timestamps[:len(kline_df)])
            sr_volume = pd.Series(kline_df['volume'].values, index=all_timestamps[:len(kline_df)])
            
            # 预测数据使用对应的时间戳
            sr_pred_close = pd.Series(pred_df['close'].values, index=pred_timestamps)
            sr_pred_volume = pd.Series(pred_df['volume'].values, index=pred_timestamps)
            
            # 为了连接预测数据和真实数据，在预测数据前添加最后一个真实数据点
            if len(sr_close) > 0:
                last_real_timestamp = sr_close.index[-1] if future_df is None else hist_timestamps.iloc[-1]
                last_real_close = sr_close.iloc[-1] if future_df is None else historical_df['close'].iloc[-1]
                last_real_volume = sr_volume.iloc[-1] if future_df is None else historical_df['volume'].iloc[-1]
                
                # 创建连接的预测数据，包含最后一个真实数据点
                pred_close_connected = pd.Series([last_real_close] + list(sr_pred_close.values), 
                                               index=[last_real_timestamp] + list(sr_pred_close.index))
                pred_volume_connected = pd.Series([last_real_volume] + list(sr_pred_volume.values), 
                                                index=[last_real_timestamp] + list(sr_pred_volume.index))
            else:
                pred_close_connected = sr_pred_close
                pred_volume_connected = sr_pred_volume
            
            sr_close.name = 'Ground Truth'
            sr_pred_close.name = "Prediction"
            sr_volume.name = 'Ground Truth'
            sr_pred_volume.name = "Prediction"
            
            # 创建图表 - 使用与 prediction_example.py 相同的尺寸
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            
            # 添加标题
            product_type_str = task.product_type.value if hasattr(task.product_type, 'value') else str(task.product_type)
            fig.suptitle(f'{task.stock_symbol} - {product_type_str}', fontsize=16, fontweight='bold')
            
            # 绘制价格 - 使用时间戳作为 x 轴
            ax1.plot(sr_close.index, sr_close.values, label='Ground Truth', color='blue', linewidth=1.5)
            ax1.plot(pred_close_connected.index, pred_close_connected.values, label='Prediction', color='red', linewidth=1.5)
            ax1.set_ylabel('Close Price', fontsize=14)
            ax1.legend(loc='lower left', fontsize=12)
            ax1.grid(True)
            
            # 绘制成交量 - 使用时间戳作为 x 轴
            ax2.plot(sr_volume.index, sr_volume.values, label='Ground Truth', color='blue', linewidth=1.5)
            ax2.plot(pred_volume_connected.index, pred_volume_connected.values, label='Prediction', color='red', linewidth=1.5)
            ax2.set_ylabel('Volume', fontsize=14)
            ax2.legend(loc='upper left', fontsize=12)
            ax2.grid(True)
            
            # 设置时间戳为横轴标签 - 自动格式化时间轴
            # matplotlib 会自动处理时间戳格式，但我们可以设置旋转角度
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            ax2.set_xlabel('Time', fontsize=14)
            
            plt.tight_layout()
            
            # 保存图片到result目录
            os.makedirs("./result", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./result/{task.stock_symbol}_task{task.id}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"预测图表已保存: {filename}")
            
        except Exception as e:
            print(f"生图失败: {str(e)}")
    
    def predict_task(self, task: Task) -> bool:
        """
        执行预测任务
        
        Args:
            task: 任务对象
            
        Returns:
            bool: 预测是否成功
        """
        try:
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            print(f"开始执行任务 {task.id}: {task.stock_symbol}")
            
            # 检查数据
            if task.historical_stock_data is None or len(task.historical_stock_data) == 0:
                raise ValueError("没有历史数据")
            
            # 加载模型
            self._load_model(task.model_name)
            
            # 准备数据 - 使用turnover字段作为amount
            x_df = task.historical_stock_data[['open', 'high', 'low', 'close', 'volume', 'turnover']].copy()
            x_df = x_df.rename(columns={'turnover': 'amount'})  # 重命名为模型期望的字段名
            
            # 从task的历史数据中获取时间戳
            x_timestamp = pd.to_datetime(task.historical_stock_data['timestamp'])
            
            # 从task的未来数据中获取预测时间戳
            y_timestamp = pd.to_datetime(task.future_stock_data['timestamp'])
            
            print(f"输入数据: {len(x_df)} 条")
            print(f"预测长度: {task.output_length} 条")
            print(f"模型参数: T={task.model_t}, top_p={task.model_p}")
            
            # 执行预测
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=len(y_timestamp), # task.output_length
                T=task.model_t,
                top_p=task.model_p,
                sample_count=10,  # 简化为单次预测
                verbose=True
            )
            
            # 保存预测结果
            task.prediction_results = pred_df
            task.status = TaskStatus.COMPLETED
            
            print(f"预测完成，生成 {len(pred_df)} 条预测数据")
            print(f"预测结果样本:")
            print(pred_df.head())
            
            # 生成图表
            self._plot_prediction(task, pred_df)
            
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            print(f"任务 {task.id} 执行失败: {str(e)}")
            return False