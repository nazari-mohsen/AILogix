from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import logging
import json
import os
from enum import Enum
from log_analyzer import LogAnalyzer, LogAnalyzerConfig
from notification_service import NotificationService, NotificationConfig

# Configure logging
logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogEntry(BaseModel):
    """Flexible log entry model that accepts any JSON payload."""
    log_data: Dict[str, Any] = Field(default_factory=dict, description="Unstructured log data")


class LogAnalysisResponse(BaseModel):
    status: str
    log_id: str
    analysis: Optional[str]
    severity: str
    similar_logs: Optional[List[Dict]]
    metrics: Optional[Dict]
    timestamp: datetime
    cache_hit: bool = Field(default=False, description="Whether the result was from cache")


app = FastAPI(
    title="Flexible Log Analysis API",
    description="API for analyzing logs in any format with ChromaDB caching",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = LogAnalyzerConfig()
log_analyzer = LogAnalyzer(config)

# Initialize NotificationService if notifications are enabled
notification_service = None
if os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true":
    notification_config = NotificationConfig(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_ids=os.getenv("TELEGRAM_CHAT_IDS", "").split(","),
        slack_webhook_urls=os.getenv("SLACK_WEBHOOK_URLS", "").split(","),
        smtp_username=os.getenv("SMTP_USERNAME"),
        smtp_password=os.getenv("SMTP_PASSWORD"),
        email_recipients=os.getenv("EMAIL_RECIPIENTS", "").split(",")
    )
    notification_service = NotificationService(notification_config)
    logger.info("Notifications are enabled.")
else:
    logger.info("Notifications are disabled.")


@app.post("/logs", response_model=LogAnalysisResponse)
async def submit_log(log_entry: LogEntry, background_tasks: BackgroundTasks, request: Request):
    """Submit a log entry for analysis with cache checking."""
    try:
        # Log the incoming request payload for debugging
        logger.info(f"Incoming request payload: {log_entry.log_data}")

        log_id = f"log_{datetime.now().timestamp()}"

        # Convert log data to a string for analysis
        log_str = json.dumps(log_entry.log_data)

        # Analyze the log
        result = log_analyzer.analyze_log(log_str)

        if result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=f"Log analysis failed: {result.get('error')}"
            )

        # Prepare analysis result for notifications
        analysis_result = {
            "severity": result["severity"],
            "timestamp": datetime.now().isoformat(),
            "analysis": result["analysis"],
            "similar_logs": result.get("similar_logs", []),
            "cache_hit": result.get("cache_hit", False)
        }

        # Send notifications in the background if enabled
        if notification_service:
            background_tasks.add_task(notification_service.notify_all, analysis_result)

        return LogAnalysisResponse(
            status="success",
            log_id=log_id,
            analysis=result["analysis"],
            severity=result["severity"],
            similar_logs=result.get("similar_logs"),
            metrics=result.get("metrics"),
            timestamp=datetime.now(),
            cache_hit=result.get("cache_hit", False)
        )

    except Exception as e:
        logger.error(f"Error processing log submission: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        count = log_analyzer.collection.count()
        return {
            "status": "success",
            "total_entries": count,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.delete("/cache")
async def clear_cache():
    """Clear the cache."""
    try:
        log_analyzer.collection.delete(where={})
        return {
            "status": "success",
            "message": "Cache cleared",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
