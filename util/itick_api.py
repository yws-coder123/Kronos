import os
import sys
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv


class _ITickKLineAPI:
    """
    内部类，用于封装iTick K线数据获取功能
    用户不应直接使用此类
    """
    
    # 不同产品类型的API基础URL
    BASE_URLS = {
        '加密货币': 'https://api.itick.org/crypto',
        '外汇': 'https://api.itick.org/forex', 
        '指数': 'https://api.itick.org/indices',
        '股票': 'https://api.itick.org/stock',
        '期货': 'https://api.itick.org/future',
        '基金': 'https://api.itick.org/fund'
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_kline_data(
        self,
        product_type: str,
        region: str,
        code: str,
        k_type: int,
        limit: int,
        et: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取K线数据

        返回例子：
        {
        "code": 0,
        "msg": null,
        "data": [
            {
            "tu": 160779.4452843,
            "c": 92490.57,
            "t": 1741239180000,
            "v": 1.7387,
            "h": 92491.67,
            "l": 92438.54,
            "o": 92474.22
            }
        ]
        }
        
        Args:
            product_type: 产品类型 (加密货币/外汇/指数/股票/期货/基金)
            region: 市场代码 (如: BA)
            code: 产品代码 (如: BTCUSDT)
            k_type: 周期类型
                1: 1分钟
                2: 5分钟
                3: 15分钟
                4: 30分钟
                5: 1小时
                8: 1天
                9: 1周
                10: 1月
            limit: 查询条数
            et: 查询截止时间 (时间戳，可选)
            
        Returns:
            Dict包含响应数据:
            - code: 响应code
            - msg: 响应描述
            - data: 响应结果数组，每个元素包含:
                - tu: 成交金额
                - c: 该K线收盘价
                - t: 时间戳
                - v: 成交数量
                - h: 该K线最高价
                - l: 该K线最低价
                - o: 该K线开盘价
        """
        # 验证产品类型
        if product_type not in self.BASE_URLS:
            raise ValueError(f"不支持的产品类型: {product_type}. 支持的类型: {list(self.BASE_URLS.keys())}")
        
        # 构建API URL
        base_url = self.BASE_URLS[product_type]
        api_url = f"{base_url}/kline"
        
        params = {
            "region": region,
            "code": code,
            "kType": k_type,
            "limit": limit
        }
        
        if et:
            params["et"] = et
            
        headers = {
            "accept": "application/json",
            "token": self.api_key
        }
        
        try:
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {str(e)}")
        except ValueError as e:
            raise Exception(f"响应数据解析失败: {str(e)}")


# 全局API实例
_api_instance: Optional[_ITickKLineAPI] = None


def _get_api_instance() -> _ITickKLineAPI:
    """获取API实例，如果不存在则创建"""
    global _api_instance
    if _api_instance is None:
        api_key = os.getenv('ITICK_API_KEY')
        if not api_key:
            raise ValueError("未找到ITICK_API_KEY环境变量，请在.env文件中配置")
        _api_instance = _ITickKLineAPI(api_key)
    return _api_instance


# 对外API接口
def get_kline_data(
    product_type: str,
    region: str,
    code: str,
    k_type: int,
    limit: int,
    et: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取K线数据
    
    Args:
        product_type: 产品类型 (加密货币/外汇/指数/股票/期货/基金)
        region: 市场代码 (如: BA)
        code: 产品代码 (如: BTCUSDT)
        k_type: 周期类型
            1: 1分钟
            2: 5分钟
            3: 15分钟
            4: 30分钟
            5: 1小时
            8: 1天
            9: 1周
            10: 1月
        limit: 查询条数
        et: 查询截止时间 (时间戳，可选)
        
    Returns:
        Dict包含响应数据:
        - code: 响应code
        - msg: 响应描述
        - data: 响应结果数组，每个元素包含:
            - tu: 成交金额
            - c: 该K线收盘价
            - t: 时间戳
            - v: 成交数量
            - h: 该K线最高价
            - l: 该K线最低价
            - o: 该K线开盘价
            
    Example:
        >>> data = get_kline_data("crypto", "BA", "BTCUSDT", 2, 10)
        >>> print(data)
    """
    api = _get_api_instance()
    return api.get_kline_data(product_type, region, code, k_type, limit, et)


def get_kline_data_with_timestamp(
    product_type: str,
    region: str,
    code: str,
    k_type: int,
    limit: int,
    end_timestamp: int
) -> Dict[str, Any]:
    """
    获取指定截止时间的K线数据
    
    Args:
        product_type: 产品类型 (加密货币/外汇/指数/股票/期货/基金)
        region: 市场代码
        code: 产品代码
        k_type: 周期类型
        limit: 查询条数
        end_timestamp: 截止时间戳（毫秒）
        
    Returns:
        同get_kline_data
    """
    return get_kline_data(product_type, region, code, k_type, limit, str(end_timestamp))


# 便捷函数 - 各产品类型专用接口
def get_crypto_kline(
    region: str,
    code: str,
    k_type: int,
    limit: int,
    et: Optional[str] = None
) -> Dict[str, Any]:
    """获取加密货币K线数据"""
    return get_kline_data('加密货币', region, code, k_type, limit, et)


def get_forex_kline(
    region: str,
    code: str,
    k_type: int,
    limit: int,
    et: Optional[str] = None
) -> Dict[str, Any]:
    """获取外汇K线数据"""
    return get_kline_data('外汇', region, code, k_type, limit, et)


def get_indices_kline(
    region: str,
    code: str,
    k_type: int,
    limit: int,
    et: Optional[str] = None
) -> Dict[str, Any]:
    """获取指数K线数据"""
    return get_kline_data('指数', region, code, k_type, limit, et)


def get_stock_kline(
    region: str,
    code: str,
    k_type: int,
    limit: int,
    et: Optional[str] = None
) -> Dict[str, Any]:
    """获取股票K线数据"""
    return get_kline_data('股票', region, code, k_type, limit, et)


def get_future_kline(
    region: str,
    code: str,
    k_type: int,
    limit: int,
    et: Optional[str] = None
) -> Dict[str, Any]:
    """获取期货K线数据"""
    return get_kline_data('期货', region, code, k_type, limit, et)


def get_fund_kline(
    region: str,
    code: str,
    k_type: int,
    limit: int,
    et: Optional[str] = None
) -> Dict[str, Any]:
    """获取基金K线数据"""
    return get_kline_data('基金', region, code, k_type, limit, et)


def main():
    """
    简单测试函数
    """
    # 设置工作目录为上一级（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    os.chdir(project_root)
    print(f"工作目录已切换到: {os.getcwd()}")
    
    # 加载环境变量
    load_dotenv()
    print("已加载.env环境变量")
    
    try:
        print("\n=== iTick API 测试 ===")
        
        # 测试获取加密货币BTCUSDT的5分钟K线数据
        print("\n正在获取加密货币BTCUSDT的5分钟K线数据...")
        data = get_crypto_kline(
            region="BA",
            code="BTCUSDT",
            k_type=2,  # 5分钟
            limit=5
        )
        
        print(f"响应状态码: {data.get('code')}")
        print(f"响应消息: {data.get('msg')}")
        
        if data.get('data'):
            print(f"\n成功获取到 {len(data['data'])} 条加密货币K线数据:")
            for i, kline in enumerate(data['data'][:3]):  # 显示前3条
                from datetime import datetime
                timestamp = datetime.fromtimestamp(kline['t'] / 1000)
                print(f"  [{i+1}] {timestamp} - 开:{kline['o']} 高:{kline['h']} 低:{kline['l']} 收:{kline['c']} 量:{kline['v']}")
        else:
            print("未获取到加密货币数据")
            
        # 测试通用接口
        print("\n\n正在测试通用接口获取外汇数据...")
        try:
            forex_data = get_kline_data(
                product_type="外汇",
                region="BA",
                code="EURUSD",
                k_type=2,  # 5分钟
                limit=3
            )
            print(f"外汇数据响应状态码: {forex_data.get('code')}")
            print(f"外汇数据响应消息: {forex_data.get('msg')}")
        except Exception as e:
            print(f"外汇数据获取失败: {str(e)}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")
        print("请检查:")
        print("1. .env文件中的ITICK_API_KEY是否正确配置")
        print("2. 网络连接是否正常")
        print("3. API密钥是否有效")


if __name__ == "__main__":
    main()