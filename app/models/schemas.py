from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class LogLevel(str, Enum):
    """Enumeration for log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationPriority(str, Enum):
    """Enumeration for notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LogEntry(BaseModel):
    """Schema for incoming log entries."""
    log_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Unstructured log data that needs to be analyzed"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Timestamp of the log entry"
    )
    source: Optional[str] = Field(
        None,
        description="Source system or application that generated the log"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "log_data": {
                    "message": "Connection timeout occurred",
                    "error_code": "E1234",
                    "service": "payment-api"
                },
                "source": "payment-service"
            }
        }


class SimilarLog(BaseModel):
    """Schema for similar log entries found in the analysis."""
    log: str = Field(..., description="The similar log entry content")
    analysis: str = Field(..., description="Analysis of the similar log")
    similarity: float = Field(..., description="Similarity score with the current log")
    severity: str = Field(..., description="Severity level of the similar log")


class MetricsSummary(BaseModel):
    """Schema for metrics summary."""
    error_patterns: Dict[str, int] = Field(
        ...,
        description="Frequency of error patterns"
    )
    metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Various metrics collected during analysis"
    )


class LogAnalysisResponse(BaseModel):
    """Schema for log analysis response."""
    status: str = Field(..., description="Status of the analysis operation")
    log_id: str = Field(..., description="Unique identifier for the log entry")
    analysis: Optional[str] = Field(None, description="Analysis results")
    severity: str = Field(..., description="Determined severity level")
    similar_logs: Optional[List[SimilarLog]] = Field(
        None,
        description="List of similar logs found"
    )
    metrics: Optional[MetricsSummary] = Field(
        None,
        description="Analysis metrics and statistics"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the analysis"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether the result was from cache"
    )


class CacheStats(BaseModel):
    """Schema for cache statistics."""
    status: str = Field(..., description="Status of the cache stats operation")
    total_entries: int = Field(..., description="Total number of entries in cache")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the stats"
    )


class NotificationRequest(BaseModel):
    """Schema for notification requests."""
    message: str = Field(..., description="Message content")
    priority: NotificationPriority = Field(..., description="Notification priority")
    channels: List[str] = Field(
        default=["slack"],
        description="List of notification channels to use"
    )
    recipients: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Optional specific recipients for each channel"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Critical system error detected",
                "priority": "high",
                "channels": ["slack", "email"],
                "recipients": {
                    "email": ["admin@example.com"],
                    "slack": ["#alerts"]
                }
            }
        }


class NotificationResponse(BaseModel):
    """Schema for notification sending results."""
    status: str = Field(..., description="Overall status of notification sending")
    results: Dict[str, bool] = Field(
        ...,
        description="Results for each notification channel"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the notification"
    )
    error: Optional[str] = Field(None, description="Error message if any")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    status: str = Field("error", description="Error status")
    detail: str = Field(..., description="Error detail message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the error"
    )


# Request Validation Models
class ClearCacheRequest(BaseModel):
    """Schema for cache clearing requests."""
    confirm: bool = Field(
        ...,
        description="Confirmation flag for clearing cache"
    )


# Response Models for API Documentation
class GenericResponse(BaseModel):
    """Schema for generic success responses."""
    status: str = Field("success", description="Operation status")
    message: str = Field(..., description="Success message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the operation"
    )
