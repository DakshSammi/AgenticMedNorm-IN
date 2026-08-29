#!/usr/bin/env python3
"""
Quota-aware LLM client for local Ollama/Qwen3 service.
Supports rate limiting, resumability, and persistent state.
"""

import json
import time
import sqlite3
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuotaAwareLLMClient:
    """Quota-aware client for local Ollama/Qwen3 service."""
    
    def __init__(self, endpoint: str, model_id: str, state_db: str,
                 max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the quota-aware client.
        
        Args:
            endpoint: Ollama API endpoint (e.g., http://localhost:11435)
            model_id: Model identifier (e.g., dengcao/Qwen3-30B-A3B-Instruct-2507:latest)
            state_db: Path to SQLite state database
            max_retries: Maximum retries for transient errors
            retry_delay: Base delay for exponential backoff
        """
        self.endpoint = endpoint.rstrip('/')
        self.model_id = model_id
        self.state_db = state_db
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize state database
        self._init_state_db()
        
        # Quota tracking
        self.quota_mode = "ADAPTIVE_UNKNOWN"  # No explicit quota discovered
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
    def _init_state_db(self):
        """Initialize the SQLite state database."""
        Path(self.state_db).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.state_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS judge_results (
                mention_id TEXT PRIMARY KEY,
                p_id TEXT,
                input_hash TEXT,
                protocol_hash TEXT,
                model_id TEXT,
                status TEXT DEFAULT 'PENDING',
                attempt_count INTEGER DEFAULT 0,
                last_attempt_at TEXT,
                next_retry_at TEXT,
                response_id TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                error_code TEXT,
                error_message_safe TEXT,
                result_path TEXT,
                result_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_status ON judge_results(status)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"State database initialized: {self.state_db}")
    
    def _get_state(self, mention_id: str) -> Optional[Dict]:
        """Get state for a mention."""
        conn = sqlite3.connect(self.state_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM judge_results WHERE mention_id = ?', (mention_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def _update_state(self, mention_id: str, **kwargs):
        """Update state for a mention."""
        conn = sqlite3.connect(self.state_db)
        cursor = conn.cursor()
        
        # Add updated_at timestamp
        kwargs['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Build update query
        set_clause = ', '.join(f'{k} = ?' for k in kwargs.keys())
        values = list(kwargs.values()) + [mention_id]
        
        cursor.execute(f'''
            UPDATE judge_results 
            SET {set_clause}
            WHERE mention_id = ?
        ''', values)
        
        conn.commit()
        conn.close()
    
    def _create_state(self, mention_id: str, p_id: str, input_hash: str, protocol_hash: str):
        """Create new state entry."""
        conn = sqlite3.connect(self.state_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO judge_results 
            (mention_id, p_id, input_hash, protocol_hash, model_id, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?)
        ''', (mention_id, p_id, input_hash, protocol_hash, self.model_id,
              datetime.now(timezone.utc).isoformat(),
              datetime.now(timezone.utc).isoformat()))
        
        conn.commit()
        conn.close()
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA256 hash of data."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def _call_ollama(self, messages: list, **kwargs) -> Dict:
        """Call Ollama API."""
        self._wait_for_rate_limit()
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 400
            }
        }
        payload.update(kwargs)
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            latency = time.time() - start_time
            self.last_request_time = time.time()
            
            result = response.json()
            
            return {
                'success': True,
                'response': result.get('message', {}).get('content', ''),
                'model': result.get('model', self.model_id),
                'latency': latency,
                'prompt_tokens': result.get('prompt_eval_count', 0),
                'completion_tokens': result.get('eval_count', 0)
            }
            
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            error_code = type(e).__name__
            
            # Check for rate limiting
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:
                    error_code = "RATE_LIMIT"
                elif e.response.status_code >= 500:
                    error_code = "SERVER_ERROR"
            
            return {
                'success': False,
                'error_code': error_code,
                'error_message': str(e)[:200],
                'latency': latency
            }
    
    def judge_medication(self, mention_id: str, p_id: str, input_data: Dict, 
                        protocol_hash: str, system_prompt: str, user_prompt: str) -> Dict:
        """
        Judge a medication mention.
        
        Returns:
            Dict with 'success' key and either 'result' or 'error' details.
        """
        input_hash = self._calculate_hash(input_data)
        
        # Check if already processed
        state = self._get_state(mention_id)
        if state and state['status'] == 'SUCCESS':
            if state['input_hash'] == input_hash and state['protocol_hash'] == protocol_hash:
                logger.info(f"Skipping {mention_id} - already processed")
                return {'success': True, 'cached': True, 'result': state}
        
        # Create or update state
        if not state:
            self._create_state(mention_id, p_id, input_hash, protocol_hash)
        else:
            self._update_state(mention_id, 
                             attempt_count=state['attempt_count'] + 1,
                             last_attempt_at=datetime.now(timezone.utc).isoformat())
        
        # Call the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(self.max_retries):
            result = self._call_ollama(messages)
            
            if result['success']:
                # Parse response
                try:
                    response_data = json.loads(result['response'])
                    
                    # Validate required fields
                    required_fields = ['mapping_assessment', 'pipeline_decision_assessment', 
                                     'confidence', 'rationale']
                    for field in required_fields:
                        if field not in response_data:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Update state to SUCCESS
                    result_hash = self._calculate_hash(response_data)
                    self._update_state(mention_id,
                                     status='SUCCESS',
                                     response_id=result.get('model', ''),
                                     input_tokens=result['prompt_tokens'],
                                     output_tokens=result['completion_tokens'],
                                     result_hash=result_hash)
                    
                    return {
                        'success': True,
                        'result': response_data,
                        'metadata': {
                            'model': result['model'],
                            'latency': result['latency'],
                            'prompt_tokens': result['prompt_tokens'],
                            'completion_tokens': result['completion_tokens']
                        }
                    }
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Invalid response for {mention_id}: {e}")
                    self._update_state(mention_id,
                                     error_code='INVALID_RESPONSE',
                                     error_message_safe=str(e)[:200])
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                    continue
            
            else:
                # Handle error
                error_code = result['error_code']
                error_msg = result['error_message']
                
                logger.warning(f"Error for {mention_id} (attempt {attempt + 1}): {error_code}")
                
                # Update state
                self._update_state(mention_id,
                                 error_code=error_code,
                                 error_message_safe=error_msg)
                
                # Check if we should retry
                if error_code in ('RATE_LIMIT', 'SERVER_ERROR', 'ConnectionError'):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                else:
                    # Permanent error
                    self._update_state(mention_id, status='FAILED_PERMANENT')
                    return {'success': False, 'error': error_msg, 'permanent': True}
        
        # All retries exhausted
        self._update_state(mention_id, status='RETRYABLE')
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        conn = sqlite3.connect(self.state_db)
        cursor = conn.cursor()
        
        stats = {}
        for status in ['PENDING', 'RUNNING', 'SUCCESS', 'RETRYABLE', 'QUOTA_WAIT', 'FAILED_PERMANENT']:
            cursor.execute('SELECT COUNT(*) FROM judge_results WHERE status = ?', (status,))
            stats[status] = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(input_tokens), SUM(output_tokens) FROM judge_results WHERE status = "SUCCESS"')
        row = cursor.fetchone()
        stats['total_input_tokens'] = row[0] or 0
        stats['total_output_tokens'] = row[1] or 0
        
        conn.close()
        return stats

def main():
    """Example usage."""
    client = QuotaAwareLLMClient(
        endpoint="http://localhost:11435",
        model_id="dengcao/Qwen3-30B-A3B-Instruct-2507:latest",
        state_db="state/qwen_judge_state.sqlite"
    )
    
    stats = client.get_stats()
    print("Current stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
