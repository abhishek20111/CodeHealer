"""
══════════════════════════════════════════════════════════════════════════════════
DATABASE TOOLS — Defined using LangChain's @tool decorator
══════════════════════════════════════════════════════════════════════════════════

NOTE: This file is IDENTICAL to ai_db_agent_langchain/tools.py.
The vector store change affects MEMORY (how history is stored/retrieved),
not TOOLS (how the agent queries databases). This is good architecture —
changing the memory layer doesn't require changing the tool layer.

INTERVIEW: "Tools are stateless functions — they don't care how their outputs
are stored. Whether the response goes to a JSON file or a vector database,
the tool just returns data. This separation of concerns makes the system modular."

KEY CONCEPT: In your MCP version, tools lived in a separate server.py and were
called over STDIO pipe. In LangChain, tools are just Python functions with the
@tool decorator. The agent calls them DIRECTLY in the same process.

PROS of LangChain tools vs MCP:
  ✅ No separate server process needed
  ✅ No STDIO pipe overhead
  ✅ Tools are just functions — easy to test, debug
  ✅ Type hints + docstrings become the tool schema automatically

CONS:
  ❌ Tools are tightly coupled to your app (MCP tools are reusable across apps)
  ❌ No protocol standard (MCP is becoming an industry standard)
  ❌ If your tool crashes, it crashes your whole app (MCP isolates in subprocess)

INTERVIEW TIP: "MCP is for tool INTEROPERABILITY (share tools across apps).
LangChain @tool is for tool SIMPLICITY (fast prototyping in one app)."
══════════════════════════════════════════════════════════════════════════════════
"""

import json
import os
from datetime import datetime
from typing import Optional

import mysql.connector
from langchain_core.tools import tool

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_SEP = "─" * 60

def log(tag: str, msg: str):
    """Structured console log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{tag}] {msg}")


# ── Database Config ──────────────────────────────────────────────────────────

DATABASES_FILE = os.path.join(os.path.dirname(__file__), "json", "databases.json")


def load_databases() -> list:
    if not os.path.exists(DATABASES_FILE):
        return []
    with open(DATABASES_FILE, "r") as f:
        return json.load(f)


def _get_connection(db_name: str):
    """
    Get a MySQL connection for the given db_name.
    
    NOTE: In production, you'd use connection pooling (e.g., SQLAlchemy engine).
    Creating a new connection per query is fine for learning but slow at scale.
    """
    dbs = load_databases()
    db_config = next((db for db in dbs if db["name"] == db_name and db.get("active")), None)
    if not db_config:
        raise ValueError(f"Database '{db_name}' not found or not active")

    return mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"],
        connection_timeout=5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — Each @tool decorated function becomes a tool the agent can call
#
# KEY CONCEPT: The DOCSTRING becomes the tool's description that the LLM sees.
# The TYPE HINTS become the input schema. LangChain auto-generates the JSON
# schema from these. This is why good docstrings matter here.
#
# INTERVIEW: "LangChain uses Pydantic under the hood to validate tool inputs.
# The @tool decorator creates a StructuredTool with an auto-generated schema."
# ══════════════════════════════════════════════════════════════════════════════


@tool
def list_databases() -> str:
    """List all active database names configured in the system.
    Use this when you need to know which databases are available."""
    log("TOOL", f"list_databases() called")
    dbs = load_databases()
    active = [db["name"] for db in dbs if db.get("active")]
    result = json.dumps(active)
    log("TOOL", f"list_databases() → {result}")
    return result


@tool
def list_tables(db_name: str) -> str:
    """List all table names in a specific database.
    Use this when you need to discover what tables exist.

    Args:
        db_name: The name of the database to list tables from.
    """
    log("TOOL", f"list_tables(db_name='{db_name}') called")
    try:
        conn = _get_connection(db_name)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        result = json.dumps(tables)
        log("TOOL", f"list_tables() → {len(tables)} tables found")
        return result
    except Exception as e:
        error = json.dumps({"error": str(e)})
        log("TOOL", f"list_tables() ERROR: {e}")
        return error


@tool
def describe_table(db_name: str, table_name: str) -> str:
    """Describe the columns, types, and keys of a specific table.
    Use this to understand the structure of a table before querying it.

    Args:
        db_name: The name of the database.
        table_name: The name of the table to describe.
    """
    log("TOOL", f"describe_table(db_name='{db_name}', table='{table_name}') called")
    try:
        conn = _get_connection(db_name)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"DESCRIBE `{table_name}`")
        columns = cursor.fetchall()

        # Get foreign keys
        cursor.execute(f"""
            SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """, (db_name, table_name))
        fks = cursor.fetchall()

        cursor.close()
        conn.close()

        result = json.dumps({
            "table": table_name,
            "columns": [
                {
                    "column_name": col["Field"],
                    "data_type": col["Type"],
                    "nullable": col["Null"],
                    "key": col["Key"],
                    "default": col["Default"],
                }
                for col in columns
            ],
            "foreign_keys": fks,
        }, default=str)
        log("TOOL", f"describe_table() → {len(columns)} columns, {len(fks)} FKs")
        return result
    except Exception as e:
        error = json.dumps({"error": str(e)})
        log("TOOL", f"describe_table() ERROR: {e}")
        return error


@tool
def run_query(db_name: str, sql: str) -> str:
    """Execute a SELECT query on a database and return the results.
    ONLY use SELECT queries. Never modify data.
    Maximum 50 rows returned.

    Args:
        db_name: The name of the database to query.
        sql: The SQL SELECT query to execute.
    """
    log("TOOL", f"run_query(db_name='{db_name}', sql='{sql[:100]}') called")

    # ── Security: Block dangerous queries ────────────────────────────────
    # INTERVIEW: This is a basic SQL injection guard. In production, use
    # parameterized queries or a read-only database user.
    blocked = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
    sql_upper = sql.upper().strip()
    for keyword in blocked:
        if keyword in sql_upper and not sql_upper.startswith("SELECT"):
            error = json.dumps({"error": f"Blocked: only SELECT queries allowed (found '{keyword}')"})
            log("TOOL", f"run_query() BLOCKED: found '{keyword}'")
            return error

    try:
        conn = _get_connection(db_name)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchmany(50)  # Limit to 50 rows
        cursor.close()
        conn.close()

        result = json.dumps(rows, default=str)
        log("TOOL", f"run_query() → {len(rows)} rows, {len(result)} chars")
        
        # Cache the query result for future semantic retrieval
        _cache_query_result(db_name, sql, result, len(rows))
        
        return result
    except Exception as e:
        error = json.dumps({"error": str(e)})
        log("TOOL", f"run_query() ERROR: {e}")
        return error


def _cache_query_result(db_name: str, sql: str, result: str, row_count: int):
    """Store query + result in ChromaDB for future semantic search."""
    try:
        from vector_store import get_collection, embed_text
        
        collection = get_collection("query_cache")
        
        # Create a rich text representation for embedding
        cache_text = f"SQL Query on {db_name}: {sql}\nResult ({row_count} rows): {result[:500]}"
        
        # Embed and store
        cache_id = f"qcache_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(sql) % 10000}"
        embedding = embed_text(cache_text)
        
        collection.add(
            ids=[cache_id],
            embeddings=[embedding],
            documents=[cache_text],
            metadatas=[{
                "db_name": db_name,
                "sql": sql[:500],
                "row_count": row_count,
                "timestamp": datetime.now().isoformat(),
                "type": "query_cache"
            }]
        )
    except Exception as e:
        # Don't fail the main query if caching fails
        log("TOOL", f"Query cache store failed (non-critical): {e}")


# ── Export all tools as a list (used by the agent) ───────────────────────────
ALL_TOOLS = [list_databases, list_tables, describe_table, run_query]
