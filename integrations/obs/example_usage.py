#!/usr/bin/env python3
"""
Example: How to use OBS Query Client

This demonstrates querying logs from OBS using the curl-based approach.
"""

import os
import sys
from dotenv import load_dotenv

# Import from same directory
from query_client import ObsQueryClient


def main():
    """Example usage of OBS Query Client"""
    
    # Load environment variables
    load_dotenv()
    
    print("=" * 70)
    print("OBS Query Client - Example Usage")
    print("=" * 70)
    
    # Initialize client (reads from .env file)
    client = ObsQueryClient()
    
    print(f"\n✅ Client initialized")
    print(f"   Proxy: {client.proxy_url}")
    print(f"   APP_ID: {client.app_id}")
    
    # Example 1: Simple search
    print(f"\n📊 Example 1: Get recent logs")
    print("-" * 70)
    
    results = client.search(time_range="1h", size=5)
    total = results.get('hits', {}).get('total', {}).get('value', 0)
    took = results.get('took', 0)
    
    print(f"Found {total} logs in last hour (query took {took}ms)")
    
    hits = results.get('hits', {}).get('hits', [])
    for i, hit in enumerate(hits, 1):
        source = hit.get('_source', {})
        timestamp = source.get('timestamp', 'N/A')
        level = source.get('level', 'N/A')
        message = source.get('message', '')[:80]
        print(f"  {i}. [{timestamp}] {level}: {message}...")
    
    # Example 2: Search for errors
    print(f"\n📊 Example 2: Search for ERROR logs")
    print("-" * 70)
    
    errors = client.get_errors(time_range="24h", size=3)
    error_count = errors.get('hits', {}).get('total', {}).get('value', 0)
    
    print(f"Found {error_count} ERROR logs in last 24 hours")
    
    error_hits = errors.get('hits', {}).get('hits', [])
    for i, hit in enumerate(error_hits, 1):
        source = hit.get('_source', {})
        timestamp = source.get('timestamp', 'N/A')
        message = source.get('message', '')[:80]
        print(f"  {i}. [{timestamp}] {message}...")
    
    # Example 3: Count logs
    print(f"\n📊 Example 3: Count logs by time range")
    print("-" * 70)
    
    count_1h = client.count(time_range="1h")
    count_24h = client.count(time_range="24h")
    count_7d = client.count(time_range="7d")
    
    print(f"  Last 1 hour:  {count_1h:,} logs")
    print(f"  Last 24 hours: {count_24h:,} logs")
    print(f"  Last 7 days:   {count_7d:,} logs")
    
    # Example 4: Query with filters
    print(f"\n📊 Example 4: Search with custom filters")
    print("-" * 70)
    
    custom_results = client.search(
        query='level:ERROR OR level:WARN',
        time_range="6h",
        size=5,
        filters=[
            {"match_phrase": {"application.name": "sampleapi"}}
        ]
    )
    
    custom_total = custom_results.get('hits', {}).get('total', {}).get('value', 0)
    print(f"Found {custom_total} ERROR/WARN logs for 'sampleapi' app in last 6 hours")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
