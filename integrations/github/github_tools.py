"""
GitHub Enterprise Tools — Read, search, and write code via GitHub API
"""

import json
import os
import base64
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_core.tools import tool

# Per-invocation file read cache to prevent infinite re-read loops
_file_read_cache = {}

# Per-turn tool-call budget so the LLM cannot loop on read/search variants.
# Reset at the start of every /ask request via _reset_file_cache().
_turn_file_reads: dict[str, int] = {}   # {"repo:path": count}
_turn_searches: int = 0
MAX_READS_PER_FILE = 2
MAX_SEARCHES_PER_TURN = 8


def _reset_file_cache():
    """Call at the start of each new agent task to clear the cache."""
    global _turn_searches
    _file_read_cache.clear()
    _turn_file_reads.clear()
    _turn_searches = 0

# ── Config ───────────────────────────────────────────────────────────────────

# Path to repos.json (in json/ folder at project root)
REPOS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "json", "repos.json")

# Dual-platform support: Enterprise (github.example.com) + Public (github.com)
GHE_HOST = os.environ.get("GHE_HOST", "github.example.com")

GITHUB_PLATFORMS = {
    GHE_HOST: {
        "api_base": f"https://{GHE_HOST}/api/v3",
        "token": os.environ.get("GHE_TOKEN", ""),
    },
    "github.com": {
        "api_base": "https://api.github.com",
        "token": os.environ.get("GITHUB_COM_TOKEN", os.environ.get("GHE_TOKEN", "")),
    },
}

def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [GITHUB] {msg}")

def _detect_platform(repo: str) -> str:
    """Detect which GitHub platform a repo belongs to.
    Supports formats:
      - 'github.com/owner/repo' → github.com
      - 'github.example.com/owner/repo' → github.example.com
      - 'owner/repo' → defaults to github.example.com (enterprise)
    """
    if repo.startswith("https://"):
        repo = repo.replace("https://", "")
    
    if repo.startswith("github.com/"):
        return "github.com"
    elif repo.startswith(f"{GHE_HOST}/"):
        return GHE_HOST
    else:
        # Default to enterprise
        return GHE_HOST

def _clean_repo(repo: str) -> str:
    """Extract 'owner/repo' from full URL or short format."""
    if repo.startswith("https://"):
        repo = repo.replace("https://", "")
    # Remove platform prefix
    for prefix in ["github.com/", f"{GHE_HOST}/"]:
        if repo.startswith(prefix):
            repo = repo[len(prefix):]
            break
    # Remove trailing slashes or paths
    parts = repo.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return repo

def _headers_for(repo: str):
    platform = _detect_platform(repo)
    token = GITHUB_PLATFORMS[platform]["token"]
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

def _api_for(repo: str, path: str) -> str:
    platform = _detect_platform(repo)
    api_base = GITHUB_PLATFORMS[platform]["api_base"]
    return f"{api_base}{path}"

# Keep backward compatibility
def _headers():
    return _headers_for("")

def _api(path: str) -> str:
    return _api_for("", path)

def load_repos() -> list:
    if not os.path.exists(REPOS_FILE):
        return []
    with open(REPOS_FILE, "r") as f:
        return json.load(f)


def _get_configured_branch(repo: str) -> Optional[str]:
    """Get the configured default branch for a repo from repos.json."""
    clean = _clean_repo(repo)
    for r in load_repos():
        r_clean = _clean_repo(r.get("repo", ""))
        if r_clean == clean:
            return r.get("branch")
    return None


def _get_upstream_repo(repo: str) -> Optional[str]:
    """Get the upstream repo for a fork from repos.json."""
    clean = _clean_repo(repo)
    for r in load_repos():
        r_clean = _clean_repo(r.get("repo", ""))
        if r_clean == clean and r.get("upstream"):
            return _clean_repo(r["upstream"])
    return None


def _resolve_default_branch(repo: str) -> str:
    """Resolve default branch: repos.json config → API fallback → 'main'."""
    configured = _get_configured_branch(repo)
    if configured:
        return configured
    # Try GitHub API to get default branch
    clean = _clean_repo(repo)
    try:
        url = _api_for(repo, f"/repos/{clean}")
        resp = requests.get(url, headers=_headers_for(repo), timeout=10)
        if resp.status_code == 200:
            return resp.json().get("default_branch", "main")
    except Exception:
        pass
    return "main"


# ══════════════════════════════════════════════════════════════════════════════
# READ-ONLY TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def list_repos() -> str:
    """List all configured GitHub repositories.
    Returns names, descriptions, and active status.
    """
    _log("list_repos() called")
    repos = load_repos()
    result = [{"name": r["name"], "repo": r["repo"], "description": r.get("description", ""), "active": r.get("active", True)} for r in repos]
    return json.dumps(result)


@tool
def read_github_file(
    repo: str,
    path: str,
    branch: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read a file from a GitHub repository (supports github.example.com and github.com).

    Prefer requesting a LINE RANGE over the whole file whenever possible — it
    keeps the LLM context small and lets you make more precise edits later.

    Args:
        repo: Repository - can be 'org/repo', 'github.com/org/repo', or 'github.example.com/org/repo'
        path: File path within the repo (e.g., 'README.md' or 'src/index.ts')
        branch: Branch name (optional, defaults to 'main')
        start_line: 1-based first line to return. If omitted, starts at line 1.
        end_line: 1-based last line to return (inclusive). If omitted and
            start_line is given, defaults to start_line + 400. If BOTH are
            omitted, the whole file is returned (truncated at 10K chars for
            backward compatibility).
    """
    clean = _clean_repo(repo)
    ref = branch or _resolve_default_branch(repo)
    range_hint = ""
    if start_line is not None or end_line is not None:
        s = max(1, start_line or 1)
        e = end_line or (s + 400)
        range_hint = f" lines={s}-{e}"
    _log(f"read_github_file(repo='{clean}', path='{path}', platform='{_detect_platform(repo)}'{range_hint})")

    # Per-turn read cap so the LLM cannot repeatedly re-read the same file
    # under the guise of different line ranges.
    read_key = f"{clean}:{path}"
    _turn_file_reads[read_key] = _turn_file_reads.get(read_key, 0) + 1
    read_n = _turn_file_reads[read_key]
    if read_n > MAX_READS_PER_FILE:
        _log(f"read_github_file() → BUDGET EXCEEDED for {path} (read #{read_n})")
        return json.dumps({
            "warning": (
                f"BUDGET EXCEEDED: you have already read '{path}' {read_n - 1} times "
                f"in this turn. Re-read denied. STOP READING — either call patch_file "
                f"to change the file, or return your findings as text. If you need a "
                f"specific line window, quote from your earlier tool results."
            ),
            "read_count": read_n - 1,
        })

    # Cache the FULL file once per (repo,path,branch); slicing happens after.
    cache_key = f"{clean}:{path}:{ref}"
    cached_full = _file_read_cache.get(cache_key)

    if cached_full is None:
        platform = _detect_platform(repo)
        if not GITHUB_PLATFORMS[platform]["token"]:
            return json.dumps({"error": f"Token not configured for {platform}"})

        url = _api_for(repo, f"/repos/{clean}/contents/{path}?ref={ref}")
        try:
            resp = requests.get(url, headers=_headers_for(repo), timeout=15)
            if resp.status_code == 404:
                return json.dumps({"error": f"File not found: {path} in {repo} (branch: {ref})"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _log(f"read_github_file() ERROR: {e}")
            return json.dumps({"error": str(e)})

        if isinstance(data, list):
            files = [{"name": f["name"], "type": f["type"], "size": f.get("size", 0)} for f in data]
            _log(f"read_github_file() → directory with {len(files)} items")
            return json.dumps({"type": "directory", "files": files})

        try:
            content = base64.b64decode(data["content"]).decode("utf-8")
        except Exception as e:
            return json.dumps({"error": f"Unable to decode file content: {e}"})

        cached_full = {"content": content, "size": data.get("size", 0)}
        _file_read_cache[cache_key] = cached_full
        _log(f"read_github_file() → {len(content)} chars (cached)")
    else:
        _log(f"read_github_file() → CACHED (already read)")

    full_content = cached_full["content"]
    total_lines = full_content.count("\n") + 1

    # Line-range mode
    if start_line is not None or end_line is not None:
        s = max(1, start_line or 1)
        e = end_line if end_line is not None else min(total_lines, s + 400)
        e = min(e, total_lines)
        if s > total_lines:
            return json.dumps({
                "file": path,
                "range": f"L{s}-L{e}",
                "total_lines": total_lines,
                "error": f"start_line {s} beyond end of file ({total_lines} lines).",
            })

        lines = full_content.splitlines()
        window = "\n".join(lines[s - 1 : e])
        result = {
            "file": path,
            "range": f"L{s}-L{e}",
            "total_lines": total_lines,
            "content": window,
        }
        span = e - s + 1
        if span > 800:
            result["warning"] = (
                f"Requested {span} lines. Consider narrowing the range or using "
                "search_github_code first — large windows waste context budget."
            )
        return json.dumps(result)

    # Whole-file mode (backward compatible)
    content_out = full_content
    if len(content_out) > 10000:
        content_out = content_out[:10000] + f"\n\n... [TRUNCATED - full file is {len(full_content)} chars, {total_lines} lines. Call with start_line/end_line to page through it.]"

    return json.dumps({
        "file": path,
        "size": cached_full.get("size", 0),
        "total_lines": total_lines,
        "content": content_out,
    })


@tool
def list_github_files(repo: str, path: Optional[str] = None, branch: Optional[str] = None) -> str:
    """List files and directories in a GitHub repository path.
    
    Args:
        repo: Repository in format 'org/repo-name'
        path: Directory path (optional, defaults to root '/')
        branch: Branch name (optional, defaults to 'main')
    """
    clean = _clean_repo(repo)
    _log(f"list_github_files(repo='{clean}', path='{path}', platform='{_detect_platform(repo)}')")
    
    platform = _detect_platform(repo)
    if not GITHUB_PLATFORMS[platform]["token"]:
        return json.dumps({"error": f"Token not configured for {platform}"})
    
    ref = branch or _resolve_default_branch(repo)
    dir_path = path or ""
    url = _api_for(repo, f"/repos/{clean}/contents/{dir_path}?ref={ref}")
    
    try:
        resp = requests.get(url, headers=_headers_for(repo), timeout=15)
        resp.raise_for_status()
        
        data = resp.json()
        if not isinstance(data, list):
            return json.dumps({"error": "Path is a file, not a directory. Use read_github_file instead."})
        
        files = [{"name": f["name"], "type": f["type"], "size": f.get("size", 0)} for f in data]
        _log(f"list_github_files() → {len(files)} items")
        return json.dumps(files)
    except Exception as e:
        _log(f"list_github_files() ERROR: {e}")
        return json.dumps({"error": str(e)})


@tool
def search_github_code(repo: str, query: str) -> str:
    """Search for code in a GitHub Enterprise repository.
    
    Args:
        repo: Repository in format 'org/repo-name'
        query: Search query (keywords, function names, error messages, etc.)
    """
    clean = _clean_repo(repo)
    _log(f"search_github_code(repo='{clean}', query='{query}', platform='{_detect_platform(repo)}')")

    global _turn_searches
    _turn_searches += 1
    if _turn_searches > MAX_SEARCHES_PER_TURN:
        _log(f"search_github_code() → BUDGET EXCEEDED (search #{_turn_searches})")
        return json.dumps({
            "warning": (
                f"BUDGET EXCEEDED: you have already run {_turn_searches - 1} code "
                f"searches in this turn. Search denied. STOP SEARCHING — commit to "
                f"a fix using patch_file, or return your findings as text based on "
                f"what you already know."
            ),
            "search_count": _turn_searches - 1,
        })

    platform = _detect_platform(repo)
    if not GITHUB_PLATFORMS[platform]["token"]:
        return json.dumps({"error": f"Token not configured for {platform}"})
    
    url = _api_for(repo, f"/search/code?q={query}+repo:{clean}&per_page=10")
    
    try:
        resp = requests.get(url, headers=_headers_for(repo), timeout=15)
        resp.raise_for_status()
        
        data = resp.json()
        results = []
        for item in data.get("items", [])[:10]:
            results.append({
                "file": item["path"],
                "url": item["html_url"],
            })
        
        # If no results and this is a fork, try upstream repo (GitHub doesn't index forks)
        if not results:
            upstream = _get_upstream_repo(repo)
            if upstream and upstream != clean:
                _log(f"search_github_code() → 0 on fork, retrying on upstream: {upstream}")
                url2 = _api_for(repo, f"/search/code?q={query}+repo:{upstream}&per_page=10")
                resp2 = requests.get(url2, headers=_headers_for(repo), timeout=15)
                if resp2.status_code == 200:
                    data = resp2.json()
                    for item in data.get("items", [])[:10]:
                        results.append({
                            "file": item["path"],
                            "url": item["html_url"],
                        })
        
        _log(f"search_github_code() → {len(results)} matches")
        return json.dumps({"total": data.get("total_count", 0), "results": results})
    except Exception as e:
        _log(f"search_github_code() ERROR: {e}")
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# WRITE TOOLS (Create branch, commit, PR)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def create_branch(repo: str, branch_name: str, from_branch: Optional[str] = None) -> str:
    """Create a new branch in a GitHub repository.

    Args:
        repo: Repository in format 'org/repo-name'
        branch_name: Name for the new branch (e.g., 'ai-fix/null-check-handler')
        from_branch: Base branch. If omitted OR the given base does not exist,
            the tool resolves the repo's default branch (repos.json → API → main).
    """
    _log(f"create_branch(repo='{repo}', branch='{branch_name}')")

    clean = _clean_repo(repo)
    base = from_branch or _resolve_default_branch(repo)

    def _try_create(base_ref: str) -> requests.Response:
        ref_url = _api_for(repo, f"/repos/{clean}/git/refs/heads/{base_ref}")
        return requests.get(ref_url, headers=_headers_for(repo), timeout=15)

    try:
        resp = _try_create(base)
        # Base branch missing on the fork — fall back to the resolved default once.
        if resp.status_code in (404, 422) and from_branch:
            default = _resolve_default_branch(repo)
            if default and default != base:
                _log(f"create_branch() base '{base}' not found (HTTP {resp.status_code}), falling back to default '{default}'")
                base = default
                resp = _try_create(base)
        resp.raise_for_status()
        sha = resp.json()["object"]["sha"]

        # Create new branch
        create_url = _api_for(repo, f"/repos/{clean}/git/refs")
        resp = requests.post(create_url, headers=_headers_for(repo), json={
            "ref": f"refs/heads/{branch_name}",
            "sha": sha,
        }, timeout=15)
        resp.raise_for_status()

        _log(f"create_branch() → created '{branch_name}' from '{base}' at {sha[:8]}")
        return json.dumps({"success": True, "branch": branch_name, "base": base, "sha": sha})
    except Exception as e:
        _log(f"create_branch() ERROR: {e}")
        return json.dumps({"error": str(e)})


@tool
def commit_file(repo: str, path: str, content: str, message: str, branch: str) -> str:
    """Create or update a file in a GitHub repository (commits directly to branch).
    
    Args:
        repo: Repository in format 'org/repo-name'
        path: File path to create/update
        content: New file content
        message: Commit message
        branch: Branch to commit to
    """
    _log(f"commit_file(repo='{repo}', path='{path}', branch='{branch}')")
    
    clean = _clean_repo(repo)
    try:
        url = _api_for(repo, f"/repos/{clean}/contents/{path}")
        
        # Check if file exists (get its SHA for update)
        existing_sha = None
        resp = requests.get(url, headers=_headers_for(repo), params={"ref": branch}, timeout=15)
        if resp.status_code == 200:
            existing_sha = resp.json().get("sha")
        
        # Create/update file
        payload = {
            "message": f"🤖 {message}",
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch,
        }
        if existing_sha:
            payload["sha"] = existing_sha
        
        resp = requests.put(url, headers=_headers_for(repo), json=payload, timeout=15)
        resp.raise_for_status()
        
        result = resp.json()
        _log(f"commit_file() → committed to {branch}")
        return json.dumps({
            "success": True,
            "commit_sha": result["commit"]["sha"],
            "url": result["content"]["html_url"],
        })
    except Exception as e:
        _log(f"commit_file() ERROR: {e}")
        return json.dumps({"error": str(e)})


@tool
def patch_file(repo: str, path: str, branch: str, find: str, replace: str, message: str) -> str:
    """Apply a find-and-replace patch to a file on a branch. Reads the full file,
    replaces the first occurrence of 'find' with 'replace', and commits.
    Use this instead of commit_file when the file is large.

    Args:
        repo: Repository in format 'org/repo-name'
        path: File path to patch (e.g., 'sampleapp-api/pom.xml')
        branch: Branch to commit to (must already exist)
        find: Exact text to find in the file (include enough context to be unique)
        replace: Text to replace it with
        message: Commit message describing the change
    """
    _log(f"patch_file(repo='{repo}', path='{path}', branch='{branch}')")
    _log(f"  find={repr(find[:80])}... replace={repr(replace[:80])}...")

    clean = _clean_repo(repo)
    try:
        # 1. Read full file content (no truncation)
        url = _api_for(repo, f"/repos/{clean}/contents/{path}")
        resp = requests.get(url, headers=_headers_for(repo), params={"ref": branch}, timeout=15)
        if resp.status_code != 200:
            return json.dumps({"error": f"Cannot read {path} on branch {branch}: HTTP {resp.status_code}"})

        data = resp.json()
        existing_sha = data["sha"]
        content = base64.b64decode(data["content"]).decode("utf-8")
        _log(f"  read {len(content)} chars, sha={existing_sha[:8]}")

        # 2. Apply patch
        if find not in content:
            return json.dumps({"error": f"Text to find not found in {path}. Make sure 'find' matches exactly."})

        patched = content.replace(find, replace, 1)
        if patched == content:
            return json.dumps({"error": "Replacement produced no change."})

        # 3. Commit
        payload = {
            "message": f"🤖 {message}",
            "content": base64.b64encode(patched.encode()).decode(),
            "sha": existing_sha,
            "branch": branch,
        }
        resp = requests.put(url, headers=_headers_for(repo), json=payload, timeout=15)
        resp.raise_for_status()

        result = resp.json()
        _log(f"patch_file() → committed to {branch}")
        return json.dumps({
            "success": True,
            "commit_sha": result["commit"]["sha"],
            "url": result["content"]["html_url"],
            "change": f"Replaced in {path}: '{find[:50]}' → '{replace[:50]}'"
        })
    except Exception as e:
        _log(f"patch_file() ERROR: {e}")
        return json.dumps({"error": str(e)})


@tool
def create_pull_request(repo: str, title: str, body: str, head_branch: str, base_branch: Optional[str] = None) -> str:
    """Create a pull request in a GitHub repository.
    
    Args:
        repo: Repository in format 'org/repo-name'
        title: PR title (e.g., '🤖 AI Fix: Null check in handler.py')
        body: PR description explaining the change
        head_branch: Source branch with changes
        base_branch: Target branch (defaults to 'main')
    """
    _log(f"create_pull_request(repo='{repo}', head='{head_branch}')")
    
    clean = _clean_repo(repo)
    base = base_branch or _resolve_default_branch(repo)
    
    try:
        url = _api_for(repo, f"/repos/{clean}/pulls")
        resp = requests.post(url, headers=_headers_for(repo), json={
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base,
        }, timeout=15)
        if resp.status_code == 422:
            detail = resp.json().get("message", "") + " " + str(resp.json().get("errors", ""))
            _log(f"create_pull_request() 422 detail: {detail}")
            return json.dumps({"error": f"422: {detail}", "base": base, "head": head_branch})
        resp.raise_for_status()
        
        pr = resp.json()
        _log(f"create_pull_request() → PR #{pr['number']}")
        return json.dumps({
            "success": True,
            "pr_number": pr["number"],
            "url": pr["html_url"],
            "state": pr["state"],
        })
    except Exception as e:
        _log(f"create_pull_request() ERROR: {e}")
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# GH CLI TOOLS — For operations that need gh/git CLI (works through SSH/SAML)
# These work even when HTTP API is blocked by firewall
# ══════════════════════════════════════════════════════════════════════════════

import subprocess
import tempfile

def _run_gh(args: list[str], repo: str = None) -> tuple[bool, str]:
    """Run a gh CLI command and return (success, output)."""
    cmd = ["gh"] + args
    env = os.environ.copy()
    
    # Clear proxy vars that block github.com connections
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        env.pop(key, None)
    
    if repo:
        clean = _clean_repo(repo)
        platform = _detect_platform(repo)
        if platform == "github.com":
            env["GH_HOST"] = "github.com"
        else:
            env["GH_HOST"] = GHE_HOST
        cmd += ["-R", clean]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip() or result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 30s"
    except Exception as e:
        return False, str(e)


@tool
def get_pr_comments(repo: str, pr_number: int) -> str:
    """Get comments on a pull request.
    
    Args:
        repo: Repository (e.g., 'github.com/your-org/smallProject' or 'github.example.com/org/repo')
        pr_number: PR number to get comments for
    """
    _log(f"get_pr_comments(repo='{repo}', pr=#{pr_number})")
    
    success, output = _run_gh(["pr", "view", str(pr_number), "--comments", "--json", "comments"], repo)
    
    if success:
        try:
            data = json.loads(output)
            comments = data.get("comments", [])
            result = []
            for c in comments:
                result.append({
                    "author": c.get("author", {}).get("login", "unknown"),
                    "body": c.get("body", "")[:500],
                    "createdAt": c.get("createdAt", ""),
                })
            _log(f"get_pr_comments() → {len(result)} comments")
            return json.dumps({"pr": pr_number, "comments": result})
        except json.JSONDecodeError:
            return json.dumps({"pr": pr_number, "raw": output[:1000]})
    else:
        _log(f"get_pr_comments() ERROR: {output}")
        return json.dumps({"error": output})


@tool
def add_pr_comment(repo: str, pr_number: int, comment: str) -> str:
    """Add a comment to a pull request.
    
    Args:
        repo: Repository (e.g., 'github.com/your-org/smallProject')
        pr_number: PR number to comment on
        comment: Comment text (supports markdown)
    """
    _log(f"add_pr_comment(repo='{repo}', pr=#{pr_number})")
    
    success, output = _run_gh(["pr", "comment", str(pr_number), "--body", comment], repo)
    
    if success:
        _log(f"add_pr_comment() → comment added")
        return json.dumps({"success": True, "message": "Comment added to PR"})
    else:
        _log(f"add_pr_comment() ERROR: {output}")
        return json.dumps({"error": output})


@tool
def check_workflow_status(repo: str, branch: Optional[str] = None) -> str:
    """Check the status of GitHub Actions workflow runs for a repo.
    
    Args:
        repo: Repository (e.g., 'github.com/your-org/smallProject')
        branch: Filter by branch (optional, shows all branches if not specified)
    """
    _log(f"check_workflow_status(repo='{repo}', branch='{branch}')")
    
    args = ["run", "list", "--limit", "5", "--json", "status,conclusion,name,headBranch,createdAt,url"]
    if branch:
        args += ["--branch", branch]
    
    success, output = _run_gh(args, repo)
    
    if success:
        try:
            runs = json.loads(output)
            result = []
            for run in runs:
                result.append({
                    "name": run.get("name", ""),
                    "status": run.get("status", ""),
                    "conclusion": run.get("conclusion", ""),
                    "branch": run.get("headBranch", ""),
                    "created": run.get("createdAt", ""),
                    "url": run.get("url", ""),
                })
            _log(f"check_workflow_status() → {len(result)} runs")
            return json.dumps({"runs": result})
        except json.JSONDecodeError:
            return json.dumps({"raw": output[:1000]})
    else:
        _log(f"check_workflow_status() ERROR: {output}")
        return json.dumps({"error": output})


@tool
def get_workflow_logs(repo: str, run_id: Optional[str] = None) -> str:
    """Get logs from a failed GitHub Actions workflow run.
    
    Args:
        repo: Repository (e.g., 'github.com/your-org/smallProject')
        run_id: Specific run ID (optional, uses latest failed run if not specified)
    """
    _log(f"get_workflow_logs(repo='{repo}', run_id='{run_id}')")
    
    if not run_id:
        # Get the latest failed run
        args = ["run", "list", "--limit", "1", "--status", "failure", "--json", "databaseId"]
        success, output = _run_gh(args, repo)
        if success:
            try:
                runs = json.loads(output)
                if runs:
                    run_id = str(runs[0]["databaseId"])
                else:
                    return json.dumps({"message": "No failed runs found"})
            except (json.JSONDecodeError, KeyError, IndexError):
                return json.dumps({"error": "Could not find failed run"})
        else:
            return json.dumps({"error": output})
    
    # Get the logs
    args = ["run", "view", run_id, "--log-failed"]
    success, output = _run_gh(args, repo)
    
    if success:
        # Truncate if too long
        if len(output) > 5000:
            output = output[-5000:]  # Last 5000 chars (most relevant)
        _log(f"get_workflow_logs() → {len(output)} chars")
        return json.dumps({"run_id": run_id, "logs": output})
    else:
        _log(f"get_workflow_logs() ERROR: {output}")
        return json.dumps({"error": output})


@tool  
def list_pull_requests(repo: str, state: Optional[str] = None) -> str:
    """List pull requests for a repository.
    
    Args:
        repo: Repository (e.g., 'github.com/your-org/smallProject')
        state: Filter by state: 'open', 'closed', 'merged', or 'all' (default: 'open')
    """
    _log(f"list_pull_requests(repo='{repo}', state='{state}')")
    
    args = ["pr", "list", "--limit", "10", "--json", "number,title,state,headRefName,author,createdAt,url"]
    if state and state != "open":
        args += ["--state", state]
    
    success, output = _run_gh(args, repo)
    
    if success:
        try:
            prs = json.loads(output)
            result = []
            for pr in prs:
                result.append({
                    "number": pr.get("number"),
                    "title": pr.get("title", ""),
                    "state": pr.get("state", ""),
                    "branch": pr.get("headRefName", ""),
                    "author": pr.get("author", {}).get("login", ""),
                    "created": pr.get("createdAt", ""),
                    "url": pr.get("url", ""),
                })
            _log(f"list_pull_requests() → {len(result)} PRs")
            return json.dumps({"prs": result})
        except json.JSONDecodeError:
            return json.dumps({"raw": output[:1000]})
    else:
        _log(f"list_pull_requests() ERROR: {output}")
        return json.dumps({"error": output})


# ── Repo map (file-summary sidecar) ─────────────────────────────────────────

from . import repo_map as _repo_map
from . import code_index as _code_index


@tool
def list_repo_map(repo: str, branch: Optional[str] = None,
                  path_filter: Optional[str] = None,
                  limit: int = 200,
                  refresh: bool = False) -> str:
    """Return a bird's-eye view of a repository — one entry per source file with
    language, symbols (top classes/functions) and a one-line summary.

    This is the FIRST tool to call before reading files. It lets you plan
    which files matter without pulling their contents. The full sidecar is
    cached under json/file_summaries/{owner}__{repo}__{branch}.json and
    refreshed only when the branch head SHA changes.

    Args:
        repo: Repository (e.g. 'your-username/sample-app').
        branch: Branch name; defaults to the repo's default branch.
        path_filter: Optional substring filter (case-insensitive) to narrow the
            listing to a directory or filename fragment (e.g. 'controller').
        limit: Max number of file entries to return (default 200).
        refresh: If True, rebuild the sidecar even if the head SHA is unchanged.
    """
    ref = branch or _resolve_default_branch(repo)
    _log(f"list_repo_map(repo='{_clean_repo(repo)}', branch='{ref}', filter='{path_filter or ''}', refresh={refresh})")
    try:
        data = _repo_map.build_repo_map(repo, ref, force=refresh, log=_log)
    except Exception as exc:
        msg = str(exc)
        # If the caller supplied a branch that does not exist, retry once
        # with the resolved default branch before giving up.
        if branch and ("404" in msg or "422" in msg):
            default = _resolve_default_branch(repo)
            if default and default != ref:
                _log(f"list_repo_map() branch '{ref}' failed ({msg[:80]}), falling back to default '{default}'")
                try:
                    data = _repo_map.build_repo_map(repo, default, force=refresh, log=_log)
                    ref = default
                except Exception as exc2:
                    _log(f"list_repo_map() fallback also failed: {exc2}")
                    return json.dumps({"error": f"branch '{branch}' not found and default '{default}' also failed: {exc2}"})
            else:
                _log(f"list_repo_map() ERROR: {exc}")
                return json.dumps({"error": str(exc)})
        else:
            _log(f"list_repo_map() ERROR: {exc}")
            return json.dumps({"error": str(exc)})
    return json.dumps(_repo_map.compact_view(data, path_filter=path_filter, limit=limit))


@tool
def index_repo_semantic(repo: str, branch: Optional[str] = None,
                        force: bool = False, limit: Optional[int] = None) -> str:
    """Build (or refresh) the semantic code index for a repository.

    Every source file listed in the repo-map sidecar is split into overlapping
    line windows, embedded with nomic-embed-text, and stored in ChromaDB. Chunks
    from files whose blob SHA is already indexed are skipped, so re-indexing on
    an unchanged branch is nearly free. This can take 30-60 s the first time —
    call once per repo, then search_repo_semantic afterwards is instant.

    Args:
        repo: Repository (e.g. 'org/repo').
        branch: Branch name; defaults to the repo's default branch.
        force: If True, drop and rebuild every file's chunks.
        limit: Optional cap on the number of files indexed this run — useful
            for a quick sanity index before committing to a full build.
    """
    ref = branch or _resolve_default_branch(repo)
    _log(f"index_repo_semantic(repo='{_clean_repo(repo)}', branch='{ref}', force={force}, limit={limit})")
    try:
        result = _code_index.index_repo(repo, ref, force=force, limit=limit, log=_log)
    except Exception as exc:
        _log(f"index_repo_semantic() ERROR: {exc}")
        return json.dumps({"error": str(exc)})
    return json.dumps(result)


@tool
def search_repo_semantic(query: str, repo: Optional[str] = None,
                         branch: Optional[str] = None, top_k: int = 8,
                         path_filter: Optional[str] = None) -> str:
    """Semantic search across the indexed code chunks of one or more repos.

    Best for concept-level queries such as "where is the authentication
    middleware", "code that handles null pointer exceptions in the payment
    flow", or "OAuth token refresh logic". Returns the top-K matching chunks
    with repo, path, line range, and a preview.

    Prerequisites:
        The target repo+branch must have been indexed with index_repo_semantic
        first. If nothing is indexed you'll get an empty list — index first.

    Args:
        query: Natural-language description of what you want to find.
        repo: Restrict search to one repo (recommended for precision).
        branch: Restrict search to one branch of that repo.
        top_k: Number of chunks to return (default 8).
        path_filter: Optional substring filter on paths (e.g. 'controller').
    """
    _log(f"search_repo_semantic(query='{query[:60]}', repo='{repo or ''}', branch='{branch or ''}', top_k={top_k})")
    try:
        hits = _code_index.search(query, repo=repo, branch=branch, top_k=top_k, path_filter=path_filter)
    except Exception as exc:
        _log(f"search_repo_semantic() ERROR: {exc}")
        return json.dumps({"error": str(exc)})
    _log(f"search_repo_semantic() → {len(hits)} hits")
    return json.dumps({
        "query": query,
        "total": len(hits),
        "hits": hits,
    })


# ── Export ────────────────────────────────────────────────────────────────────

GITHUB_READ_TOOLS = [
    list_repos, read_github_file, list_github_files, search_github_code,
    list_repo_map, index_repo_semantic, search_repo_semantic,
]
GITHUB_WRITE_TOOLS = [create_branch, commit_file, patch_file, create_pull_request]
GITHUB_CLI_TOOLS = [get_pr_comments, add_pr_comment, check_workflow_status, get_workflow_logs, list_pull_requests]
ALL_GITHUB_TOOLS = GITHUB_READ_TOOLS + GITHUB_WRITE_TOOLS + GITHUB_CLI_TOOLS
