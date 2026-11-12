#!/usr/bin/env python3
"""
OBS Query Client - Production Implementation
Uses curl subprocess approach (bypasses local proxy blocking)

Based on:
- curl approach for internal APIs behind restrictive proxies
- OBS Proxy Service documentation
- Successful testing with curl

Purpose: Query logs FROM OBS (not send logs TO OBS)
Service: obsproxy-*.example.com (Query API)
"""

import subprocess
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class ObsQueryClient:
    """
    Client for querying logs from OBS using Proxy Service API.
    
    Uses curl subprocess to bypass local proxy User-Agent filtering.
    Useful for internal APIs behind restrictive proxies.
    """
    
    def __init__(
        self,
        proxy_url: Optional[str] = None,
        app_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize OBS Query Client.
        
        Args:
            proxy_url: OBS Proxy URL (default: from env OBS_PROXY_URL)
            app_id: Application APP_ID (default: from env OBS_APP_ID)
            username: Username for basic auth (default: from env OBS_USERNAME)
            password: Password for basic auth (default: from env OBS_PASSWORD)
            timeout: Request timeout in seconds
        """
        # Helper to strip quotes from env values
        def clean_env(value: Optional[str]) -> Optional[str]:
            if value:
                return value.strip("'\"")
            return value
        
        self.proxy_url = clean_env(proxy_url or os.getenv("OBS_PROXY_URL", "https://obsproxy-qa.example.com"))
        self.app_id = clean_env(app_id or os.getenv("OBS_APP_ID"))
        self.username = clean_env(username or os.getenv("OBS_USERNAME"))
        self.password = clean_env(password or os.getenv("OBS_PASSWORD"))
        self.timeout = timeout
        
        if not self.app_id:
            raise ValueError("APP_ID must be provided (via parameter or OBS_APP_ID env var)")
        if not self.username or not self.password:
            raise ValueError("Username and password must be provided")
    
    def _execute_query(
        self,
        index_pattern: str,
        query_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute query using curl subprocess.
        
        Args:
            index_pattern: OpenSearch index pattern
            query_body: OpenSearch query DSL
            
        Returns:
            dict: OpenSearch response
            
        Raises:
            Exception: If curl command fails
        """
        url = f"{self.proxy_url}/{index_pattern}/_search"

        # Debug trace so we can compare against the Slack/OpenSearch link.
        from datetime import datetime
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [OBS] HTTP POST {url}", flush=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [OBS] body={json.dumps(query_body)[:600]}", flush=True)
        
        # Use curl with --noproxy to bypass local proxy blocking
        # Local proxy at localhost blocks CONNECT tunnel with 403
        # TLS verification is on by default; set OBS_INSECURE_SSL=true only for self-signed internal certs.
        insecure = os.getenv("OBS_INSECURE_SSL", "false").lower() == "true"
        result = subprocess.run([
            "curl",
            "--noproxy", "*",  # Bypass local proxy (critical!)
            "-X", "POST",
            url,
            "-u", f"{self.username}:{self.password}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(query_body),
            *(["-k"] if insecure else []),
            "-s",  # Silent mode (no progress bar)
        ], capture_output=True, text=True, timeout=self.timeout)
        
        if result.returncode != 0:
            raise Exception(
                f"curl command failed:\n"
                f"Return code: {result.returncode}\n"
                f"STDERR: {result.stderr}\n"
                f"STDOUT: {result.stdout[:500]}"
            )
        
        if not result.stdout.strip():
            raise Exception(f"Empty response from curl. STDERR: {result.stderr}")
        
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise Exception(
                f"Invalid JSON response:\n"
                f"Error: {str(e)}\n"
                f"Response: {result.stdout[:500]}"
            )
        total = parsed.get('hits', {}).get('total', {}).get('value', 0)
        took = parsed.get('took', 0)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [OBS] HTTP response OK: total={total}, took={took}ms, response_size={len(result.stdout)}B", flush=True)
        return parsed
    
    def search(
        self,
        query: Optional[str] = None,
        time_range: str = "1h",
        size: int = 100,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        filters: Optional[List[Dict[str, Any]]] = None,
        cluster_prefix: str = "*",
    ) -> Dict[str, Any]:
        """
        Search OBS logs.
        
        Args:
            query: Query string (optional)
            time_range: Time range (e.g., "1h", "24h", "7d")
            size: Number of results (max 1000 per OBS rules)
            sort_by: Field to sort by
            sort_order: "asc" or "desc"
            filters: Additional OpenSearch filters
            cluster_prefix: Cluster prefix (e.g., "mobile", "*")
            
        Returns:
            dict: Search results
        """
        # Validate size
        if size > 1000:
            raise ValueError("OBS allows max 1000 docs per request")
        
        # Build index pattern
        index_pattern = f"{cluster_prefix}:obs-{self.app_id}-*"
        
        # Build query
        must_clauses = []
        filter_clauses = []
        
        # Add time range (required by OBS)
        filter_clauses.append(self._parse_time_range(time_range))
        
        # Add query string if provided
        if query:
            must_clauses.append({
                "query_string": {"query": query}
            })
        
        # Add additional filters
        if filters:
            filter_clauses.extend(filters)
        
        query_body = {
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}],
                    "filter": filter_clauses
                }
            },
            "size": size,
            "sort": [{sort_by: {"order": sort_order}}],
            "_source": True,
            # Ask OpenSearch for the REAL total, not the 10 000 cap.
            "track_total_hits": True,
        }
        
        return self._execute_query(index_pattern, query_body)
    
    def _parse_time_range(self, time_range: str) -> Dict[str, Any]:
        """Parse time range string to OpenSearch query.

        OpenSearch date math accepts INTEGER units only (y/M/w/d/h/m/s).
        Round fractional values up so `7.5h` becomes `8h` instead of
        silently returning zero hits.
        """
        import re as _re
        m = _re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([yMwdhms])\s*", time_range or "")
        if m:
            value = float(m.group(1))
            unit = m.group(2)
            if value != int(value):
                new_value = int(value) + 1
                from datetime import datetime as _dt
                print(f"[{_dt.now().strftime('%H:%M:%S')}] [OBS] normalised time_range {time_range!r} → {new_value}{unit} (OpenSearch date math needs integers)", flush=True)
                time_range = f"{new_value}{unit}"

        return {
            "range": {
                "timestamp": {
                    "gte": f"now-{time_range}",
                    "lte": "now"
                }
            }
        }
    
    def get_errors(
        self,
        application_name: Optional[str] = None,
        time_range: str = "24h",
        size: int = 100,
    ) -> Dict[str, Any]:
        """Get ERROR level logs."""
        filters = []
        
        if application_name:
            filters.append({
                "match_phrase": {"application.name": application_name}
            })
        
        filters.append({
            "match_phrase": {"level": "ERROR"}
        })
        
        return self.search(
            time_range=time_range,
            size=size,
            filters=filters
        )
    
    def count(
        self,
        query: Optional[str] = None,
        time_range: str = "24h",
        filters: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Count logs matching query."""
        result = self.search(
            query=query,
            time_range=time_range,
            size=0,  # We only need count
            filters=filters
        )
        return result.get('hits', {}).get('total', {}).get('value', 0)


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    client = ObsQueryClient()
    
    print("Searching logs from last hour...")
    results = client.search(time_range="1h", size=5)
    
    total = results.get('hits', {}).get('total', {}).get('value', 0)
    print(f"Total hits: {total}")
    
    hits = results.get('hits', {}).get('hits', [])
    for i, hit in enumerate(hits, 1):
        source = hit.get('_source', {})
        print(f"\n{i}. {source.get('timestamp')} - {source.get('level')} - {source.get('message', '')[:100]}")
