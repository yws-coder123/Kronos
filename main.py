from datetime import datetime, timedelta, time, timezone

from task_generator import TaskGenerator
from util.task_model import TimeGranularity, ProductType
from get_stock_data.itick_data_fetcher import ITickDataFetcher
from run_kronos.kronos_predictor import KronosTaskPredictor


def main(i=0):
    """主函数：生成任务列表示例"""
    # 创建任务生成器
    generator = TaskGenerator()
    
    # 定义股票列表
    stock_list = ['BTCUSDT']
    
    # 生成任务列表
    tasks=[]
    for i in range(0, 90, 1):
        _tasks = generator.generate_from_stock_list(
            stock_symbols=stock_list,
            start_prediction_timestamp=datetime(2025, 8, 15, 12, 0, 0, 0) - timedelta(hours=i),
            time_granularity=TimeGranularity.HOUR_1,
            product_type=ProductType.CRYPTO,
            input_length=32,
            output_length=4,
        )
        tasks.extend(_tasks)
    
    # 打印生成的任务
    print(f"生成了 {len(tasks)} 个任务:")
    for task in tasks:
        print(f"任务ID: {task.id}, 产品类型: {task.product_type}, 股票: {task.stock_symbol}, 时间粒度: {task.time_granularity}")
    
    # 获取股票数据
    fetcher = ITickDataFetcher()
    predictor = KronosTaskPredictor(enable_plotting=True)
    try:
        print("\n开始获取股票数据...")
        
        for task in tasks:
            print(f"\n处理任务 {task.id}: {task.stock_symbol}")
            fetcher.fetch_and_process_data(task)

            historical_count = len(task.historical_stock_data) if task.historical_stock_data is not None else 0
            print(f"  历史数据: {historical_count} 条")
            if historical_count != task.input_length:
                print(f"  ⚠️ 警告: 历史数据条数({historical_count})与输入长度({task.input_length})不一致")
            future_count = len(task.future_stock_data) if task.future_stock_data is not None else 0
            print(f"  未来数据: {future_count} 条")
            if future_count != task.output_length:
                print(f"  ⚠️ 警告: 未来数据条数({future_count})与输出长度({task.output_length})不一致")

            # 测试预测模块
            print("\n开始测试预测模块...")
            if task.historical_stock_data is not None and len(task.historical_stock_data) > 0:
                print(f"\n测试任务 {task.id}: {task.stock_symbol}")
                success = predictor.predict_task(task)

                if success:
                    print(f"✅ 任务 {task.id} 预测成功")
                    print(f"预测状态: {task.status}")
                    if task.prediction_results is not None:
                        print(f"预测结果条数: {len(task.prediction_results)}")
                else:
                    print(f"❌ 任务 {task.id} 预测失败")

        else:
            print("没有找到有效的历史数据，跳过预测测试")


    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        print("请确保已设置ITICK_API_KEY环境变量")
        return tasks
    

    
    return tasks


if __name__ == "__main__":
    main()