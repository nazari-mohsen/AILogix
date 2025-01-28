from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends, status
from datetime import datetime
import logging
from app.models.schemas import (
    LogEntry,
    LogAnalysisResponse,
    NotificationRequest,
    NotificationResponse,
    CacheStats,
    GenericResponse,
)
from app.services.log_analyzer import LogAnalyzer, LogAnalyzerConfig
from app.services.notification import NotificationService
from app.core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize services
log_analyzer = LogAnalyzer(LogAnalyzerConfig())

# Initialize NotificationService if enabled
notification_service = None
settings = get_settings()
if settings.ENABLE_NOTIFICATIONS:
    notification_service = NotificationService()
    logger.info("Notification service initialized")


# Dependency for notification service
async def get_notification_service():
    if not notification_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Notification service is not enabled"
        )
    return notification_service


@router.post("/logs", response_model=LogAnalysisResponse)
async def submit_log(
    log_entry: LogEntry,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Submit a log entry for analysis.
    - Analyzes the log content using AI
    - Checks for similar patterns in history
    - Sends notifications if configured
    """
    try:
        logger.info(f"Processing log entry: {log_entry.log_data}")

        # Generate unique log ID
        log_id = f"log_{datetime.now().timestamp()}"

        # Analyze the log
        analysis_result = log_analyzer.analyze_log(str(log_entry.log_data))

        if analysis_result["status"] != "success":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Log analysis failed: {analysis_result.get('error')}"
            )

        # Prepare analysis result for notifications
        notification_data = {
            "severity": analysis_result["severity"],
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis_result["analysis"],
            "similar_logs": analysis_result.get("similar_logs", []),
            "cache_hit": analysis_result.get("cache_hit", False)
        }

        # Send notifications in background if service is enabled
        if notification_service:
            background_tasks.add_task(
                notification_service.notify_all,
                notification_data
            )

        return LogAnalysisResponse(
            status="success",
            log_id=log_id,
            analysis=analysis_result["analysis"],
            severity=analysis_result["severity"],
            similar_logs=analysis_result.get("similar_logs"),
            metrics=analysis_result.get("metrics"),
            timestamp=datetime.now(),
            cache_hit=analysis_result.get("cache_hit", False)
        )

    except Exception as e:
        logger.error(f"Error processing log submission: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/notify", response_model=NotificationResponse)
async def send_notification(
    notification: NotificationRequest,
    notification_service: NotificationService = Depends(get_notification_service)
):
    """
    Send a notification through configured channels.
    - Supports multiple notification channels
    - Handles priority-based routing
    """
    try:
        notification_data = {
            "severity": notification.priority.value,
            "timestamp": datetime.now().isoformat(),
            "analysis": notification.message
        }

        results = await notification_service.notify_all(notification_data)
        return NotificationResponse(
            status="success",
            results=results,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """Get statistics about the analysis cache."""
    try:
        stats = {
            "total_entries": log_analyzer.collection.count()
        }
        return CacheStats(
            status="success",
            total_entries=stats["total_entries"],
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error retrieving cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/cache", response_model=GenericResponse)
async def clear_cache():
    """Clear the analysis cache."""
    try:
        log_analyzer.collection.delete(where={})
        return GenericResponse(
            status="success",
            message="Cache cleared successfully",
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", response_model=GenericResponse)
async def health_check():
    """Check the health status of the API and its services."""
    try:
        # Check core services
        cache_count = log_analyzer.collection.count()
        services_status = {
            "log_analyzer": True,
            "cache": cache_count >= 0,
            "notifications": notification_service is not None
        }

        # If all services are operational, return success
        if all(services_status.values()):
            return GenericResponse(
                status="success",
                message="All services operational",
                timestamp=datetime.now()
            )
        else:
            raise Exception("Some services are down")

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service health check failed: {str(e)}"
        )
