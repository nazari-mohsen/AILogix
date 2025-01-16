from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import aiohttp
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class NotificationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationConfig:
    # Telegram settings
    telegram_bot_token: Optional[str] = None
    telegram_chat_ids: List[str] = None

    # Slack settings
    slack_webhook_urls: List[str] = None
    slack_channel: Optional[str] = "#monitoring"

    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = None

    # Notification thresholds
    min_severity_telegram: str = "warning"
    min_severity_slack: str = "info"
    min_severity_email: str = "error"


class NotificationFormatter:
    """Formats notifications for different platforms."""

    @staticmethod
    def format_telegram(analysis_result: Dict) -> str:
        """Format message for Telegram."""
        return (
            f"🔍 *Log Analysis Alert*\n\n"
            f"*Severity:* {analysis_result['severity']}\n"
            f"*Timestamp:* {analysis_result['timestamp']}\n\n"
            f"*Analysis:*\n{analysis_result['analysis']}\n\n"
            f"*Similar Incidents:* {len(analysis_result.get('similar_logs', []))}\n"
            f"*Cache Hit:* {analysis_result.get('cache_hit', False)}"
        )

    @staticmethod
    def format_slack(analysis_result: Dict) -> Dict:
        """Format message for Slack."""
        color = {
            "debug": "#808080",
            "info": "#36a64f",
            "warning": "#ffcc00",
            "error": "#ff0000",
            "critical": "#9b0000"
        }.get(analysis_result['severity'].lower(), "#36a64f")

        return {
            "attachments": [
                {
                    "color": color,
                    "title": "Log Analysis Alert",
                    "fields": [
                        {
                            "title": "Severity",
                            "value": analysis_result['severity'],
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": analysis_result['timestamp'],
                            "short": True
                        },
                        {
                            "title": "Analysis",
                            "value": analysis_result['analysis'],
                            "short": False
                        },
                        {
                            "title": "Similar Incidents",
                            "value": str(len(analysis_result.get('similar_logs', []))),
                            "short": True
                        },
                        {
                            "title": "Cache Hit",
                            "value": str(analysis_result.get('cache_hit', False)),
                            "short": True
                        }
                    ]
                }
            ]
        }

    @staticmethod
    def format_email(analysis_result: Dict) -> tuple:
        """Format message for email."""
        subject = f"Log Analysis Alert - {analysis_result['severity'].upper()}"

        body = f"""
        <html>
        <body>
            <h2>Log Analysis Alert</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; text-align: left;">Severity</th>
                    <td style="padding: 8px;">{analysis_result['severity']}</td>
                </tr>
                <tr>
                    <th style="padding: 8px; text-align: left;">Timestamp</th>
                    <td style="padding: 8px;">{analysis_result['timestamp']}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; text-align: left;">Analysis</th>
                    <td style="padding: 8px;">{analysis_result['analysis']}</td>
                </tr>
                <tr>
                    <th style="padding: 8px; text-align: left;">Similar Incidents</th>
                    <td style="padding: 8px;">{len(analysis_result.get('similar_logs', []))}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; text-align: left;">Cache Hit</th>
                    <td style="padding: 8px;">{analysis_result.get('cache_hit', False)}</td>
                </tr>
            </table>
        </body>
        </html>
        """

        return subject, body


class NotificationService:
    """Handles sending notifications to multiple platforms."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.formatter = NotificationFormatter()

    async def send_telegram(self, analysis_result: Dict) -> bool:
        """Send notification via Telegram."""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_ids:
            return False

        if analysis_result['severity'].lower() not in ['warning', 'error', 'critical']:
            return False

        message = self.formatter.format_telegram(analysis_result)
        success = True

        async with aiohttp.ClientSession() as session:
            for chat_id in self.config.telegram_chat_ids:
                try:
                    url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
                    params = {
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "Markdown"
                    }

                    async with session.post(url, json=params) as response:
                        if response.status != 200:
                            logger.error(f"Failed to send Telegram message: {await response.text()}")
                            success = False

                except Exception as e:
                    logger.error(f"Error sending Telegram notification: {e}")
                    success = False

        return success

    async def send_slack(self, analysis_result: Dict) -> bool:
        """Send notification via Slack."""
        if not self.config.slack_webhook_urls:
            return False

        message = self.formatter.format_slack(analysis_result)
        success = True

        async with aiohttp.ClientSession() as session:
            for webhook_url in self.config.slack_webhook_urls:
                try:
                    async with session.post(webhook_url, json=message) as response:
                        if response.status != 200:
                            logger.error(f"Failed to send Slack message: {await response.text()}")
                            success = False

                except Exception as e:
                    logger.error(f"Error sending Slack notification: {e}")
                    success = False

        return success

    def send_email(self, analysis_result: Dict) -> bool:
        """Send notification via email."""
        if not all([
            self.config.smtp_username,
            self.config.smtp_password,
            self.config.email_recipients
        ]):
            return False

        if analysis_result['severity'].lower() not in ['error', 'critical']:
            return False

        try:
            subject, body = self.formatter.format_email(analysis_result)

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.email_recipients)

            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False

    async def notify_all(self, analysis_result: Dict) -> Dict[str, bool]:
        """Send notifications to all configured platforms."""
        results = {}

        # Send to all platforms concurrently
        tasks = []
        if self.config.telegram_bot_token:
            tasks.append(self.send_telegram(analysis_result))
        if self.config.slack_webhook_urls:
            tasks.append(self.send_slack(analysis_result))

        # Execute async tasks
        if tasks:
            notification_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            results['telegram'] = notification_results[0] if self.config.telegram_bot_token else False
            results['slack'] = notification_results[1] if self.config.slack_webhook_urls else False

        # Send email (synchronous)
        results['email'] = self.send_email(analysis_result)

        return results


if __name__ == "__main__":
    # Example configuration
    config = NotificationConfig(
        telegram_bot_token="YOUR_BOT_TOKEN",
        telegram_chat_ids=["CHAT_ID1", "CHAT_ID2"],
        slack_webhook_urls=["YOUR_WEBHOOK_URL"],
        smtp_username="your-email@gmail.com",
        smtp_password="your-app-password",
        email_recipients=["recipient@example.com"]
    )

    # Example analysis result
    analysis_result = {
        "severity": "error",
        "timestamp": datetime.now().isoformat(),
        "analysis": "Database connection timeout detected. Multiple retry attempts failed.",
        "similar_logs": [{"id": "1"}, {"id": "2"}],
        "cache_hit": False
    }

    # Initialize service
    notification_service = NotificationService(config)

    # Send notifications
    async def main():
        results = await notification_service.notify_all(analysis_result)
        print("Notification results:", results)

    asyncio.run(main())
