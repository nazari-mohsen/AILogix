# AILogix - Intelligent Log Analysis & Solutions

![AILogix Image](https://github.com/nazari-mohsen/AILogix/blob/main/images/AILogix.png)

**AILogix** is a powerful, AI-driven log analysis tool designed to help you understand, diagnose, and solve issues within your system logs. By leveraging advanced machine learning and natural language processing techniques, AILogix provides deep insights into your logs, detects error patterns, and offers actionable solutions, all in real-time.
Whether you're managing a large-scale production environment or working on smaller projects, AILogix helps you quickly identify and address potential issues before they escalate, reducing downtime and improving system reliability.

## Why AILogix?

- **AI-Powered Insights**: AILogix uses state-of-the-art AI models to analyze your logs intelligently, identifying error patterns, anomalies, and trends that might go unnoticed by traditional methods.
- **Automated Problem Solving**: Not only does AILogix identify issues, but it also offers potential solutions, helping you take immediate action to resolve problems and optimize system performance.
- **Severity Assessment**: AILogix assesses the severity of errors in real-time (Debug, Info, Warning, Error, Critical), providing you with a clear picture of your system’s health.
- **Enhanced Log Management**: With powerful storage and search capabilities via ChromaDB, AILogix helps you efficiently manage and retrieve logs for further analysis.
- **Actionable Alerts & Notifications**: AILogix integrates with platforms like Telegram, Slack, and Email to send real-time notifications, ensuring you're always informed of critical issues.

AILogix is designed for developers, system administrators, and DevOps teams who need a smarter, more efficient way to manage log data, automate troubleshooting, and ensure the reliability of their systems.

## Features

- **AI-Powered Log Insights**:  
  AILogix utilizes state-of-the-art AI models to intelligently analyze logs, detecting error patterns, anomalies, and trends that may be overlooked by traditional methods.  
  Powered by [Ollama](https://ollama.com), AILogix allows you to easily switch between different models for log analysis by modifying the `.env` configuration file. This flexibility enables you to customize the analysis to suit your needs.

- **Automated Problem Solving**:  
  AILogix doesn't just identify issues – it also offers potential solutions, helping you resolve problems quickly and optimize system performance.

- **Real-time Severity Assessment**:  
  AILogix assesses the severity of each log entry (Debug, Info, Warning, Error, Critical), helping you understand the current health of your system at a glance.

- **Advanced Log Management**:  
  With robust storage and search functionalities powered by ChromaDB, AILogix helps you manage, retrieve, and analyze logs effortlessly.

- **Log Pattern Detection**:  
  AILogix detects similar patterns across logs to enhance analysis, allowing for a more accurate diagnosis of recurring issues.

- **Metrics Reporting and Management**:  
  AILogix tracks various metrics related to log entries, helping you monitor the performance and health of your system over time.

- **Real-time Notifications**:  
  AILogix can send alerts to different platforms to keep you informed of critical issues:
  - **Telegram**: Send messages to a Telegram group or channel to notify about issues or log analysis.
  - **Slack**: Get notified in a Slack channel for alerts and system status updates.
  - **Email**: Receive email notifications with alert messages or detailed reports.

> **Note:** To enable notifications (Telegram, Slack, Email), set `ENABLE_NOTIFICATIONS=true` in your `.env` file.

## Installation and Setup

### 1. Clone the Repository

First, clone the project repository:

```bash
git clone https://github.com/nazari-mohsen/AILogix
cd AILogix

```
### 2. Set Up Environment Using Docker Compose

#### 2.1. Build and Start Containers

To build and start the project using Docker Compose, run the following command:
```bash
make up
```

This command will start the project using Docker Compose and run the containers in the background.
#### 2.2. Stop and Remove Containers

To stop and remove the containers:
```bash
make down
```

#### 2.3. Build Docker Images

If you have made changes to the Dockerfile or need to rebuild the Docker images, use the following command:
```bash
make build
```
#### 2.4. Clean (Remove Containers, Networks, and Images)

To clean up the containers, networks, and images:
```bash
make clean
```
#### 2.5. View Logs

To view Docker logs:
```bash
make logs
```
### 3. Pull Models for Log Analysis

To pull a specific model for log analysis, you can use the following command:
```bash
make pull_model model=llama3.2
```
Make sure to replace llama3.2 with the name of the model you want to pull. You can specify any model compatible with the Ollama platform.

### 4. Sending Logs

In this section, we will explain how to send logs.

### 4.1. Sending Logs Using `curl`

To send a log, you can use the `curl` command. Below is an example of how to send a log to the server:

```bash
curl -X POST http://localhost:8000/logs -H "Content-Type: application/json" -d '{"timestamp": "2025-01-16T12:00:00Z", "level": "ERROR", "message": "An error occurred while processing the request."}'
‍
```
## 4.2. Managing Cache

In this section, we will explain how to use different endpoints to manage the cache.

### 4.2.1. Getting Cache Statistics Using `GET /cache/stats`

You can use this endpoint to retrieve statistics about the current cache in the system. This typically includes details such as the cache size, the number of cached items, and other relevant cache status information.

#### `curl` Command to Get Cache Statistics:

```bash
curl -X GET http://localhost:8000/cache/stats

```
### 4.2.2. Clearing Cache Using `DELETE /cache`

You can use this endpoint to clear the current cache. This operation is typically performed when you want to free up cache memory or when you need to reload the data.

#### `curl` Command to Clear Cache:

```bash
curl -X DELETE http://localhost:8000/cache
```
Support
-------

For support or to report issues, please open a new [**Issue**](https://github.com/nazari-mohsen/AILogix/issues) on GitHub Issues.

License
-------

This project is licensed under the MIT License.
