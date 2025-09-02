"""
Pydantic models for task configuration management.
"""

from enum import Enum
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd


class TaskStatus(str, Enum):
    """Task execution status enumeration."""
    NOT_RUN = "not_run"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProductType(str, Enum):
    """Product type enumeration based on iTick API."""
    CRYPTO = "加密货币"
    FOREX = "外汇"
    INDICES = "指数"
    STOCK = "股票"
    FUTURE = "期货"
    FUND = "基金"


class TimeGranularity(str, Enum):
    # Specific minute intervals
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"

    # Specific hour intervals
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"

    # Specific day intervals
    DAY_1 = "1d"
    DAY_3 = "3d"

    # Specific week intervals
    WEEK_1 = "1w"
    WEEK_2 = "2w"

    # Specific month intervals
    MONTH_1 = "1M"
    MONTH_3 = "3M"


class Task(BaseModel):
    """
    Individual task configuration with validation.

    Each task represents a single stock symbol processing job
    with specific parameters and execution status.
    """

    id: int = Field(..., ge=1, description="Sequential task ID starting from 1")
    status: TaskStatus = Field(TaskStatus.NOT_RUN, description="Task execution status")
    product_type: ProductType = Field(..., description="Product type for data source")
    stock_symbol: str = Field(..., descripion="Stock symbol (e.g., AAPL, GOOGL)")
    start_prediction_timestamp: datetime = Field(..., description="Start prediction timestamp")
    time_granularity: TimeGranularity = Field(..., description="Time granularity for data")

    # Model parameters
    model_name: str = Field("Kronos-base", description="Model name or URL")
    model_p: float = Field(0.9, ge=0.0, le=1.0, description="Model parameter p")
    model_t: float = Field(0.6, ge=0.0, le=1.0, description="Model parameter t")

    # Input/Output lengths (output is 1/4 of input)
    input_length: int = Field(512, ge=8, description="Model input length")
    output_length: Optional[int] = Field(None, description="Model output length (1/4 of input)")

    # Execution parameters
    run_count: int = Field(10, ge=1, description="Number of runs with different seeds")
    
    # Data storage
    historical_stock_data: Optional[Any] = Field(None, description="Stock data before start_prediction_timestamp")
    future_stock_data: Optional[Any] = Field(None, description="Stock data after start_prediction_timestamp")
    
    # Prediction results
    prediction_results: Optional[Any] = Field(None, description="Model prediction results")

    @field_validator('stock_symbol')
    @classmethod
    def validate_stock_symbol(cls, v):
        """Validate stock symbol format."""
        if not v or not v.strip():
            raise ValueError('Stock symbol cannot be empty')
        return v.upper().strip()


    @model_validator(mode='after')
    def set_output_length(self):
        """If output_length is not provided, it is automatically calculated as 1/4 of input_length."""
        if self.output_length is None:
            self.output_length = self.input_length // 4
        return self

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True

