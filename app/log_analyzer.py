from dataclasses import dataclass
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb import HttpClient
from datetime import datetime
from typing import List, Dict, Optional, TypedDict, Any
import numpy as np
import logging
import json
import re
import os
from enum import Enum

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log_analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class LogSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogResponse(TypedDict):
    status: str
    log_entry: str
    analysis: Optional[str]
    similar_logs: Optional[List[Dict]]
    metrics: Optional[Dict]
    error: Optional[str]
    severity: Optional[str]


@dataclass
class LogAnalyzerConfig:
    model_name: str = os.getenv("MODEL_NAME", "llama3.2")
    collection_name: str = os.getenv("COLLECTION_NAME", "log_history")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.92"))
    max_history_entries: int = int(os.getenv("MAX_HISTORY_ENTRIES", "100"))
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8001"))
    ollama_host: str = os.getenv("OLLAMA_HOST", "localhost")
    ollama_port: int = int(os.getenv("OLLAMA_PORT", "11434"))
    analysis_window: int = int(os.getenv("ANALYSIS_WINDOW", "20"))
    alert_threshold: float = float(os.getenv("ALERT_THRESHOLD", "0.85"))
    retention_days: int = int(os.getenv("RETENTION_DAYS", "30"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))


class MetricsCollector:
    """Collects and manages log analysis metrics."""

    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.alert_history: List[Dict] = []
        self.error_patterns: Dict[str, int] = {}

    def record_metric(self, name: str, value: Any, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })

    def track_error_pattern(self, error_signature: str):
        """Track recurring error patterns."""
        self.error_patterns[error_signature] = self.error_patterns.get(error_signature, 0) + 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Generate a summary of current metrics."""
        summary = {
            'error_patterns': dict(sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'metrics': {}
        }

        for name, values in self.metrics.items():
            if values:
                recent_values = [v['value'] for v in values[-10:]]
                summary['metrics'][name] = {
                    'current': values[-1]['value'],
                    'average': np.mean(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values)
                }
        return summary


class LogAnalysisContext:
    """Manages log analysis context and pattern detection."""

    def __init__(self):
        self.recent_logs: List[Dict] = []
        self.pattern_history: Dict[str, List[Dict]] = {}
        self.metrics_collector = MetricsCollector()

    def update_context(self, log_entry: str, analysis: str, severity: LogSeverity) -> None:
        """Update analysis context with new log entry."""
        timestamp = datetime.now()

        entry = {
            'log': log_entry,
            'analysis': analysis,
            'timestamp': timestamp.isoformat(),
            'severity': severity.value,
            'metrics': self.metrics_collector.get_metrics_summary()
        }

        self.recent_logs.append(entry)

        # Maintain fixed window of recent logs
        if len(self.recent_logs) > 100:
            self.recent_logs.pop(0)

        # Track patterns for similar types of logs
        pattern_key = self._extract_pattern(log_entry)
        if pattern_key not in self.pattern_history:
            self.pattern_history[pattern_key] = []
        self.pattern_history[pattern_key].append(entry)

    def _extract_pattern(self, log_entry: str) -> str:
        """Extract pattern signature from log entry."""
        # Example: Extract error codes or key phrases
        error_code_match = re.search(r'error code: (\d+)', log_entry, re.IGNORECASE)
        if error_code_match:
            return f"error_code_{error_code_match.group(1)}"

        # Fallback to a generic pattern
        return "generic_pattern"

    def get_relevant_context(self, log_entry: str) -> Dict:
        """Get relevant context for log analysis."""
        pattern_key = self._extract_pattern(log_entry)
        similar_patterns = self.pattern_history.get(pattern_key, [])

        return {
            'recent_logs': self.recent_logs[-20:],
            'similar_patterns': similar_patterns[-5:],
            'metrics': self.metrics_collector.get_metrics_summary()
        }


class LogAnalyzer:
    """Advanced log analysis system with pattern detection and similarity matching."""

    def __init__(self, config: LogAnalyzerConfig):
        self.config = config
        self.context = LogAnalysisContext()
        self._initialize_components()
        self._setup_analysis_chain()

    def _initialize_components(self) -> None:
        """Initialize core components with error handling."""
        try:
            # Connect to remote ChromaDB
            self.chroma_client = HttpClient(host=self.config.chroma_host, port=self.config.chroma_port)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Connected to ChromaDB at {self.config.chroma_host}:{self.config.chroma_port}")

            # Initialize Ollama components
            self.llm = OllamaLLM(
                model=self.config.model_name,
                temperature=0.7,
                base_url=f"http://{self.config.ollama_host}:{self.config.ollama_port}"
            )

            self.embeddings = OllamaEmbeddings(
                model=self.config.model_name,
                base_url=f"http://{self.config.ollama_host}:{self.config.ollama_port}"
            )
            logger.info(f"Initialized Ollama at {self.config.ollama_host}:{self.config.ollama_port}")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _setup_analysis_chain(self) -> None:
        """Set up log analysis prompt chain."""
        analysis_prompt = """
        Log Entry: {log_entry}
        Recent Context: {context}

        Analyze the log entry and provide:
        1. Severity assessment
        2. Root cause analysis
        3. Potential impact
        4. Recommended actions
        5. Pattern correlation with recent logs

        If the log entry is unclear or needs more context, specify what additional information would be helpful.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert log analysis system."),
            ("human", analysis_prompt)
        ])

        self.chain = (
            {
                "log_entry": RunnablePassthrough(),
                "context": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def analyze_log(self, log_entry: str) -> LogResponse:
        """Analyze log entry with enhanced pattern detection."""
        try:
            # Generate embedding for the log entry
            log_embedding = self.embeddings.embed_query(log_entry)
            logger.info(f"Generated embedding for log entry: {log_embedding[:5]}...")

            # Get relevant context
            context = self.context.get_relevant_context(log_entry)

            # Find similar logs
            similar_logs = self._find_similar_logs(log_embedding)
            logger.info(f"Found {len(similar_logs)} similar logs.")

            # Determine if this is a cache hit
            cache_hit = len(similar_logs) > 0

            # Generate analysis (only if no cache hit)
            analysis = None
            if not cache_hit:
                analysis = self.chain.invoke({
                    "log_entry": log_entry,
                    "context": json.dumps(context, indent=2)
                })
                logger.info("Generated new analysis using Ollama.")
            else:
                logger.info("Cache hit: Using existing analysis from similar logs.")
                analysis = similar_logs[0]['analysis']  # Use the analysis from the most similar log

            # Determine severity
            severity = self._assess_severity(log_entry, analysis)

            # Update context
            self.context.update_context(log_entry, analysis, severity)

            # Store log entry (only if no cache hit)
            if not cache_hit:
                self._store_log(log_entry, analysis, log_embedding, severity)
                logger.info("Stored log entry in ChromaDB.")

            return LogResponse(
                status="success",
                log_entry=log_entry,
                analysis=analysis,
                similar_logs=similar_logs,
                metrics=context['metrics'],
                severity=severity.value,
                cache_hit=cache_hit,
                error=None
            )

        except Exception as e:
            logger.error(f"Error analyzing log: {e}")
            return LogResponse(
                status="error",
                log_entry=log_entry,
                analysis=None,
                similar_logs=None,
                metrics=None,
                severity=None,
                cache_hit=False,
                error=str(e)
            )

    def _assess_severity(self, log_entry: str, analysis: str) -> LogSeverity:
        """Assess log severity based on content and analysis."""
        if "critical" in log_entry.lower() or "fatal" in log_entry.lower():
            return LogSeverity.CRITICAL
        elif "error" in log_entry.lower():
            return LogSeverity.ERROR
        elif "warning" in log_entry.lower():
            return LogSeverity.WARNING
        elif "info" in log_entry.lower():
            return LogSeverity.INFO
        else:
            return LogSeverity.DEBUG

    def _find_similar_logs(self, log_embedding: List[float]) -> List[Dict]:
        """Find similar log entries."""
        try:
            collection_count = self.collection.count()
            logger.info(f"Number of logs in collection: {collection_count}")

            if collection_count == 0:
                logger.info("No logs in the collection.")
                return []

            # Query the collection
            results = self.collection.query(
                query_embeddings=[log_embedding],
                n_results=5,
                include=['metadatas', 'documents', 'distances']
            )

            similar_logs = []
            for doc, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                similarity = 1 - distance  # Convert distance to similarity
                if similarity >= self.config.similarity_threshold:
                    similar_logs.append({
                        'log': doc,
                        'analysis': meta['analysis'],
                        'similarity': similarity,
                        'severity': meta['severity']
                    })

            logger.info(f"Similar logs found: {similar_logs}")
            return similar_logs

        except Exception as e:
            logger.error(f"Error finding similar logs: {e}")
            return []

    def _store_log(self, log_entry: str, analysis: str, embedding: List[float], severity: LogSeverity) -> None:
        """Store log entry with analysis results."""
        try:
            self.collection.add(
                documents=[log_entry],
                embeddings=[embedding],
                metadatas=[{
                    'analysis': analysis,
                    'severity': severity.value,
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[str(datetime.now().timestamp())]
            )
            logger.info(f"Stored log entry in ChromaDB: {log_entry}")
        except Exception as e:
            logger.error(f"Failed to store log: {e}")
