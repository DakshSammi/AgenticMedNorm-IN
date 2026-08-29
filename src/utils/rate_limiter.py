"""
Rate limiter for managing token-per-minute (TPM) and requests-per-minute (RPM) budgets.

The internal qwen3-27b API has hard limits:
  - 500K tokens per minute
  - 120 requests per minute

This module provides a rolling-window rate limiter that tracks token usage
and intelligently spaces out requests to stay within these limits.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


logger = logging.getLogger("RateLimiter")


@dataclass
class TokenUsageRecord:
    """Track a single API request's token usage."""
    timestamp: float
    document_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated: bool = False  # Whether values are estimated or from API
    reason: str = ""  # "rate_limit", "timeout", etc.
    
    def age_seconds(self) -> float:
        """Return age of this record in seconds."""
        return time.time() - self.timestamp


class RollingWindowRateLimiter:
    """
    Tracks token usage in a rolling 60-second window.
    
    Provides intelligent backoff when approaching rate limits.
    """
    
    def __init__(
        self,
        tpm_limit: int = 500000,
        rpm_limit: int = 120,
        window_seconds: int = 60,
        buffer_seconds: int = 15,
        max_retries_rate_limit: int = 1,
    ):
        """
        Args:
            tpm_limit: Tokens per minute limit (default 500K)
            rpm_limit: Requests per minute limit (default 120)
            window_seconds: Rolling window size (default 60 sec)
            buffer_seconds: Safety buffer before window expiry (default 15 sec)
            max_retries_rate_limit: Max retries for rate-limit errors (default 1)
        """
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.window_seconds = window_seconds
        self.buffer_seconds = buffer_seconds
        self.max_retries_rate_limit = max_retries_rate_limit
        
        self.usage_history: List[TokenUsageRecord] = []
        self.retry_count: Dict[str, int] = {}  # Per document ID
        
    def _prune_old_records(self) -> None:
        """Remove records older than window_seconds."""
        cutoff_time = time.time() - self.window_seconds
        self.usage_history = [
            r for r in self.usage_history
            if r.timestamp > cutoff_time
        ]
    
    def get_rolling_tokens(self) -> int:
        """Get total tokens used in current rolling window."""
        self._prune_old_records()
        return sum(r.total_tokens for r in self.usage_history)
    
    def get_rolling_requests(self) -> int:
        """Get total requests in current rolling window."""
        self._prune_old_records()
        return len(self.usage_history)
    
    def get_time_to_available_tokens(self, needed_tokens: int) -> float:
        """
        Calculate seconds to wait until 'needed_tokens' become available.
        
        Returns: float >= 0 (wait time in seconds)
        """
        self._prune_old_records()
        
        current_tokens = self.get_rolling_tokens()
        available = self.tpm_limit - current_tokens
        
        if available >= needed_tokens:
            return 0.0
        
        # Find the oldest record; once it ages past window_seconds, tokens free up
        if not self.usage_history:
            return 0.0
        
        oldest = min(self.usage_history, key=lambda r: r.timestamp)
        wait_time = (oldest.timestamp + self.window_seconds + self.buffer_seconds) - time.time()
        
        return max(0.0, wait_time)
    
    def record_usage(
        self,
        document_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated: bool = False,
        reason: str = "",
    ) -> None:
        """Record a completed API request."""
        total = prompt_tokens + completion_tokens
        record = TokenUsageRecord(
            timestamp=time.time(),
            document_id=document_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated=estimated,
            reason=reason,
        )
        self.usage_history.append(record)
        self._prune_old_records()
        
        logger.info(
            f"Recorded usage for {document_id}: "
            f"{prompt_tokens} prompt + {completion_tokens} completion = {total} total "
            f"({'estimated' if estimated else 'from API'}). "
            f"Rolling total: {self.get_rolling_tokens()}/{self.tpm_limit} TPM"
        )
    
    def estimate_token_components(
        self,
        document_id: str,
        max_tokens: int,
        num_images: int,
        compressed_image_size_kb: Optional[float] = None,
        prompt_length_chars: Optional[int] = None,
        reserve_full_output_budget: bool = False,
    ) -> Dict[str, int]:
        """
        Estimate token pressure for a document.

        Conservative fallback mode reserves the full max_tokens output budget.
        Use that mode when the API may reserve TPM based on requested output
        capacity or when actual usage is unavailable.
        """
        visual_tokens = num_images * 10
        if compressed_image_size_kb:
            visual_tokens += int(compressed_image_size_kb * 0.8)

        prompt_tokens = 500  # Baseline for system + user prompt
        if prompt_length_chars:
            prompt_tokens += int(prompt_length_chars / 4)

        output_budget = max_tokens if reserve_full_output_budget else int(max_tokens * 0.2)
        total_estimate = visual_tokens + prompt_tokens + output_budget

        logger.info(
            f"Estimated tokens for {document_id}: "
            f"visual={visual_tokens}, prompt={prompt_tokens}, output_budget={output_budget} "
            f"({'full max_tokens reservation' if reserve_full_output_budget else '20% output estimate'}) "
            f"-> total={total_estimate}"
        )

        return {
            "visual_tokens": visual_tokens,
            "prompt_tokens": prompt_tokens,
            "input_tokens": visual_tokens + prompt_tokens,
            "output_budget_tokens": output_budget,
            "total_tokens": total_estimate,
        }

    def estimate_tokens(
        self,
        document_id: str,
        max_tokens: int,
        num_images: int,
        compressed_image_size_kb: Optional[float] = None,
        prompt_length_chars: Optional[int] = None,
        reserve_full_output_budget: bool = False,
    ) -> int:
        """Estimate total token pressure for a document."""
        return self.estimate_token_components(
            document_id=document_id,
            max_tokens=max_tokens,
            num_images=num_images,
            compressed_image_size_kb=compressed_image_size_kb,
            prompt_length_chars=prompt_length_chars,
            reserve_full_output_budget=reserve_full_output_budget,
        )["total_tokens"]
    
    def should_wait_before_request(
        self,
        document_id: str,
        estimated_tokens: int,
    ) -> tuple[bool, float]:
        """
        Check if we should wait before making the next request.
        
        Returns: (should_wait: bool, wait_seconds: float)
        """
        self._prune_old_records()
        
        current_tokens = self.get_rolling_tokens()
        current_requests = self.get_rolling_requests()
        
        # Check TPM limit
        if current_tokens + estimated_tokens > self.tpm_limit:
            wait_time = self.get_time_to_available_tokens(estimated_tokens)
            logger.warning(
                f"TPM limit would be exceeded. "
                f"Current: {current_tokens}/{self.tpm_limit}, "
                f"Need: {estimated_tokens}. Wait: {wait_time:.1f} sec"
            )
            return True, wait_time
        
        # Check RPM limit
        if current_requests >= self.rpm_limit:
            oldest = min(self.usage_history, key=lambda r: r.timestamp)
            wait_time = (oldest.timestamp + self.window_seconds) - time.time()
            logger.warning(
                f"RPM limit would be exceeded. "
                f"Current: {current_requests}/{self.rpm_limit}. Wait: {max(0, wait_time):.1f} sec"
            )
            return True, max(0.0, wait_time)
        
        return False, 0.0
    
    def wait_if_needed(
        self,
        document_id: str,
        estimated_tokens: int,
    ) -> float:
        """
        Sleep if rate limit would be exceeded. Return actual sleep time.
        """
        should_wait, wait_time = self.should_wait_before_request(document_id, estimated_tokens)
        
        if should_wait:
            # Add a small buffer
            adjusted_wait = wait_time + self.buffer_seconds
            logger.info(f"Rate limit: sleeping {adjusted_wait:.1f} sec before {document_id}")
            time.sleep(adjusted_wait)
            return adjusted_wait
        
        return 0.0
    
    def should_retry_after_rate_limit_error(self, document_id: str) -> bool:
        """Check if we should retry this document after a rate-limit error."""
        count = self.retry_count.get(document_id, 0)
        if count < self.max_retries_rate_limit:
            self.retry_count[document_id] = count + 1
            return True
        return False
    
    def reset_retry_count(self, document_id: str) -> None:
        """Reset retry counter for a document (after success)."""
        self.retry_count[document_id] = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current rate limiter state summary."""
        self._prune_old_records()
        return {
            "rolling_tokens": self.get_rolling_tokens(),
            "tpm_limit": self.tpm_limit,
            "tpm_remaining": self.tpm_limit - self.get_rolling_tokens(),
            "rolling_requests": self.get_rolling_requests(),
            "rpm_limit": self.rpm_limit,
            "rpm_remaining": self.rpm_limit - self.get_rolling_requests(),
            "window_seconds": self.window_seconds,
            "usage_records": len(self.usage_history),
        }
    
    def format_summary_for_log(self) -> str:
        """Format summary as a single log line."""
        summary = self.get_summary()
        return (
            f"Rate Limiter: {summary['rolling_tokens']}/{summary['tpm_limit']} TPM, "
            f"{summary['rolling_requests']}/{summary['rpm_limit']} RPM"
        )


# Singleton instance for global access
_global_limiter: Optional[RollingWindowRateLimiter] = None


def get_global_limiter() -> RollingWindowRateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RollingWindowRateLimiter()
    return _global_limiter


def init_global_limiter(
    tpm_limit: int = 500000,
    rpm_limit: int = 120,
    window_seconds: int = 60,
    buffer_seconds: int = 15,
    max_retries_rate_limit: int = 1,
) -> RollingWindowRateLimiter:
    """Initialize the global rate limiter with custom settings."""
    global _global_limiter
    _global_limiter = RollingWindowRateLimiter(
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit,
        window_seconds=window_seconds,
        buffer_seconds=buffer_seconds,
        max_retries_rate_limit=max_retries_rate_limit,
    )
    return _global_limiter
