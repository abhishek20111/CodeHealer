"""
LangChain Tools for OBS Integration
Wraps OBS Query Client for use with LangChain agents
"""

import os
import json as _json
from typing import Optional, List
from langchain_core.tools import tool

# Import the query client
try:
    from .query_client import ObsQueryClient
except ImportError:
    from query_client import ObsQueryClient


def _obs_log(msg: str) -> None:
    """Print OBS debug line with timestamp so we can trace exactly what was sent."""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [OBS] {msg}", flush=True)


def create_obs_tools() -> List:
    """
    Create LangChain tools for OBS integration.
    
    Returns:
        List of decorated tool functions
        
    Raises:
        ValueError: If OBS credentials not configured
    """
    # Initialize OBS client
    try:
        client = ObsQueryClient()
    except ValueError as e:
        # Credentials not configured
        raise ValueError(f"OBS not configured: {e}")
    
    @tool
    def search_obs_logs(query: Optional[str] = None, time_range: str = "1h", size: int = 10) -> str:
        """Search application logs in OBS (log observability).

        Correct field names (nested — use dot notation):
          - application.name  (NOT application_name)
          - level             (values: ERROR, WARN, INFO, DEBUG)
          - message           (free text of the log line)

        Example valid queries:
          - 'level:ERROR AND application.name:sampleapi'
          - 'message:"NoResourceFoundException"'

        NOTE: For simple "errors for app X" questions, prefer the structured
        `get_obs_errors(application_name=..., time_range=...)` tool — it's more
        reliable than a free-text query.

        Time ranges: '1h', '2h', '5h', '24h', '7d'
        """
        _obs_log(f"search_obs_logs called: query={query!r}, time_range={time_range!r}, size={size}")
        try:
            results = client.search(
                query=query,
                time_range=time_range,
                size=min(size, 100)  # Limit to 100 for agent responses
            )
            
            total = results.get('hits', {}).get('total', {}).get('value', 0)
            took = results.get('took', 0)
            hits = results.get('hits', {}).get('hits', [])
            _obs_log(f"search_obs_logs → total={total}, took={took}ms, returned_hits={len(hits)}")

            # Auto-retry a 0-hit result. If broader queries reveal real data,
            # transparently fall back and return those hits so the caller does
            # not need a second round trip.
            verify_note = ""
            if total == 0:
                _obs_log(f"search_obs_logs → 0 hits. Verifying with query=None, same time_range={time_range}…")
                try:
                    verify = client.search(query=None, time_range=time_range, size=min(size, 20))
                    verify_total = verify.get('hits', {}).get('total', {}).get('value', 0)
                    verify_hits = verify.get('hits', {}).get('hits', [])
                    _obs_log(f"search_obs_logs → verify (no query) → total={verify_total}")
                    if verify_total > 0 and query:
                        verify_note = (
                            f"\n⚠️ AUTO-RETRIED: 0 hits for your query={query!r} in {time_range}, but "
                            f"{verify_total:,} logs exist without that query. Falling back to unfiltered "
                            f"results below so you have actual data to look at. Your query string likely "
                            f"used the wrong field name — inspect a sample and re-filter (e.g. use "
                            f"structured field like `application.name` or the get_obs_errors tool).\n"
                        )
                        total = verify_total
                        hits = verify_hits
                    elif verify_total == 0:
                        wide = client.search(query=None, time_range="24h", size=min(size, 20))
                        wide_total = wide.get('hits', {}).get('total', {}).get('value', 0)
                        wide_hits = wide.get('hits', {}).get('hits', [])
                        _obs_log(f"search_obs_logs → verify 24h → total={wide_total}")
                        if wide_total > 0:
                            verify_note = (
                                f"\n⚠️ AUTO-RETRIED: 0 hits in {time_range}, but {wide_total:,} logs exist "
                                f"in the last 24h. Falling back to the 24h window below.\n"
                            )
                            total = wide_total
                            hits = wide_hits
                except Exception as ve:
                    _obs_log(f"search_obs_logs → verify FAILED: {ve}")

            # Format response
            response = f"Found {total:,} logs (query took {took}ms)\n\n"
            if verify_note:
                response += verify_note + "\n"
            
            if hits:
                response += "Sample results:\n"
                for i, hit in enumerate(hits[:size], 1):
                    source = hit.get('_source', {})
                    timestamp = source.get('timestamp', 'N/A')
                    level = source.get('level', 'N/A')
                    message = source.get('message', '')[:200]
                    app_name = source.get('application', {}).get('name', 'N/A')
                    
                    response += f"\n{i}. [{timestamp}] {level} - {app_name}\n"
                    response += f"   {message}\n"
            else:
                response += "No logs found in the specified time range."
            
            return response
            
        except Exception as e:
            _obs_log(f"search_obs_logs ERROR: {e}")
            return f"Error querying OBS: {str(e)}"
    
    @tool
    def get_obs_errors(application_name: Optional[str] = None, time_range: str = "24h", size: int = 10) -> str:
        """Get ERROR level logs from OBS for an application.

        PREFER THIS over search_obs_logs when the user asks about errors for a
        specific application — it uses structured filters (level:ERROR +
        application.name:<app>) instead of free-text queries, so it can't be
        fooled by field-name typos.

        `application_name` is a plain app id (e.g. 'sampleapi', 'app-batch'),
        NOT an OpenSearch query string.

        Time ranges: '1h', '2h', '5h', '24h', '7d'
        Size: how many actual hits to include in the response (max 50).

        Response includes total count + up to `size` timestamped entries.
        Do NOT call this tool again for the same app + range; drill down with
        get_pr_comments / GitHub tools or a single search_obs_logs with a
        `message:"..."` filter instead.
        """
        _obs_log(f"get_obs_errors called: application_name={application_name!r}, time_range={time_range!r}, size={size}")
        try:
            results = client.get_errors(
                application_name=application_name,
                time_range=time_range,
                size=min(size, 50)
            )
            
            total = results.get('hits', {}).get('total', {}).get('value', 0)
            hits = results.get('hits', {}).get('hits', [])
            _obs_log(f"get_obs_errors → total={total}, returned_hits={len(hits)}")

            # Auto-retry a 0-hit result. If a broader query reveals real errors,
            # transparently fall back and return those hits so the caller does
            # not need a second round trip.
            verify_note = ""
            if total == 0:
                if application_name:
                    _obs_log(f"get_obs_errors → 0 hits. Verifying WITHOUT app filter (same time_range={time_range})…")
                    try:
                        verify = client.get_errors(application_name=None, time_range=time_range, size=min(size, 20))
                        verify_total = verify.get('hits', {}).get('total', {}).get('value', 0)
                        verify_hits = verify.get('hits', {}).get('hits', [])
                        sample_apps = sorted({(h.get('_source', {}).get('application') or {}).get('name', '?') for h in verify_hits})
                        _obs_log(f"get_obs_errors → verify (no app filter) → total={verify_total}, sample_apps={sample_apps}")
                        if verify_total > 0:
                            verify_note = (
                                f"\n⚠️ AUTO-RETRIED: 0 hits for application_name={application_name!r} in {time_range}, but "
                                f"{verify_total:,} ERROR logs exist in the same window without the app filter "
                                f"(sample apps actually present: {sample_apps}). "
                                f"Falling back to unfiltered results below. Check the app name (case / prefix) "
                                f"and re-call with the correct value if `{application_name}` was wrong.\n"
                            )
                            total = verify_total
                            hits = verify_hits
                    except Exception as ve:
                        _obs_log(f"get_obs_errors → verify without app filter FAILED: {ve}")
                else:
                    _obs_log(f"get_obs_errors → 0 hits (no app filter). Verifying with wider time_range=24h…")
                    try:
                        verify = client.get_errors(application_name=None, time_range="24h", size=min(size, 20))
                        verify_total = verify.get('hits', {}).get('total', {}).get('value', 0)
                        verify_hits = verify.get('hits', {}).get('hits', [])
                        _obs_log(f"get_obs_errors → verify (24h) → total={verify_total}")
                        if verify_total > 0:
                            verify_note = (
                                f"\n⚠️ AUTO-RETRIED: 0 hits in {time_range} but {verify_total:,} ERROR logs exist "
                                f"in the last 24h. Falling back to the 24h window below.\n"
                            )
                            total = verify_total
                            hits = verify_hits
                    except Exception as ve:
                        _obs_log(f"get_obs_errors → verify 24h FAILED: {ve}")

            response = f"Found {total:,} ERROR logs in last {time_range}\n\n"
            if verify_note:
                response += verify_note + "\n"

            if hits:
                for i, hit in enumerate(hits[:size], 1):
                    source = hit.get('_source', {})
                    timestamp = source.get('timestamp', 'N/A')
                    message = source.get('message', '')[:300]
                    app_name = source.get('application', {}).get('name', 'N/A')
                    
                    response += f"\n{i}. [{timestamp}] {app_name}\n"
                    response += f"   {message}\n"
            else:
                response += "No ERROR logs found."
            
            return response
            
        except Exception as e:
            _obs_log(f"get_obs_errors ERROR: {e}")
            return f"Error querying OBS: {str(e)}"
    
    @tool
    def count_obs_logs(query: Optional[str] = None, time_range: str = "24h") -> str:
        """Count logs in OBS matching criteria.

        Correct field names (nested — use dot notation):
          - application.name  (NOT application_name)
          - level             (values: ERROR, WARN, INFO, DEBUG)

        Example valid query: 'level:ERROR AND application.name:sampleapi'

        Time ranges: '1h', '2h', '5h', '24h', '7d'
        """
        _obs_log(f"count_obs_logs called: query={query!r}, time_range={time_range!r}")
        try:
            count = client.count(query=query, time_range=time_range)
            _obs_log(f"count_obs_logs → {count}")

            if count == 0 and query:
                _obs_log(f"count_obs_logs → 0 hits. Verifying with query=None, same time_range={time_range}…")
                try:
                    verify_count = client.count(query=None, time_range=time_range)
                    _obs_log(f"count_obs_logs → verify (no query) → total={verify_count}")
                    if verify_count > 0:
                        return (
                            f"Found 0 logs in last {time_range} for query={query!r}.\n"
                            f"⚠️ AUTO-VERIFIED: {verify_count:,} logs exist in the same window without the query. "
                            f"Your query string returned nothing — check field names (use `application.name`, "
                            f"not `application_name` or `application`). Or drop the query and use the "
                            f"structured `get_obs_errors(application_name=..., time_range=...)` tool.\n"
                        )
                except Exception as ve:
                    _obs_log(f"count_obs_logs → verify FAILED: {ve}")

            return f"Found {count:,} logs in last {time_range}"
        except Exception as e:
            _obs_log(f"count_obs_logs ERROR: {e}")
            return f"Error counting logs: {str(e)}"
    
    # Return list of tools
    return [search_obs_logs, get_obs_errors, count_obs_logs]


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Creating OBS tools...")
    try:
        tools = create_obs_tools()
        print(f"✅ Created {len(tools)} OBS tools")
        
        for tool in tools:
            print(f"\n📦 {tool.name}")
            print(f"   {tool.description}")
        
        # Test search
        print("\n🧪 Testing search_obs_logs...")
        result = tools[0].invoke({"time_range": "1h", "size": 2})
        print(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
