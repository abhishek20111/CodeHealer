"""File-summary sidecar generator for GitHub repos.

For each repo+branch we build a compact "repo map":
    {
        "repo": "org/name",
        "branch": "production-api",
        "head_sha": "...",
        "generated_at": "ISO8601",
        "total_files": N,
        "files": [
            {
                "path": "src/main/java/.../SampleAppController.java",
                "lang": "java",
                "lines": 245,
                "sha": "…",
                "summary": "first comment or docstring",
                "symbols": ["SampleAppController", "getSummary", ...]
            },
            ...
        ]
    }

The sidecar is saved under json/file_summaries/{owner}__{repo}__{branch}.json
so the LLM can be given a bird's-eye view of the codebase without paying an
API round-trip per file on every question.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import requests

# ── Config ───────────────────────────────────────────────────────────────────

# Cap the number of files we FETCH content for (symbol extraction).
# Files beyond this cap are still listed in the map with path+lang only.
MAX_FILES_TO_FETCH = int(os.environ.get("REPO_MAP_MAX_FETCH", "300"))

# Skip files larger than this many bytes — usually generated / minified / data
MAX_FILE_BYTES = int(os.environ.get("REPO_MAP_MAX_FILE_BYTES", "200000"))

# Parallel workers for content fetching
FETCH_WORKERS = int(os.environ.get("REPO_MAP_WORKERS", "6"))

_SOURCE_EXTS = {
    "py": "python", "pyi": "python",
    "java": "java", "kt": "kotlin", "scala": "scala",
    "js": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "go": "go", "rs": "rust", "rb": "ruby",
    "c": "c", "h": "c", "cc": "cpp", "cpp": "cpp", "hpp": "cpp",
    "cs": "csharp", "vb": "vb",
    "php": "php", "swift": "swift", "m": "objc",
    "sh": "shell", "bash": "shell", "zsh": "shell",
    "sql": "sql",
    "yaml": "yaml", "yml": "yaml",
    "xml": "xml", "json": "json",
    "toml": "toml", "ini": "ini",
    "md": "markdown", "rst": "rst",
    "dockerfile": "dockerfile",
}

# Config files worth including even without an extension
_CONFIG_FILENAMES = {
    "Dockerfile", "Makefile", "Jenkinsfile", "Procfile",
    "pom.xml", "build.gradle", "settings.gradle",
    "package.json", "requirements.txt", "setup.py", "pyproject.toml",
    "go.mod", "Cargo.toml", "Gemfile",
    ".env.example", "README.md",
}

_SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env",
    ".gradle", "target", "build", "dist", ".next", "out",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "coverage", ".idea", ".vscode",
}

_lock = threading.Lock()


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class FileEntry:
    path: str
    lang: str
    lines: int = 0
    sha: str = ""
    summary: str = ""
    symbols: list[str] = field(default_factory=list)
    size: int = 0


# ── Sidecar path ────────────────────────────────────────────────────────────

def sidecar_path(repo: str, branch: str) -> str:
    safe = repo.replace("/", "__")
    branch_safe = re.sub(r"[^A-Za-z0-9._-]", "_", branch)
    base = os.path.join(os.path.dirname(__file__), "..", "..", "json", "file_summaries")
    os.makedirs(base, exist_ok=True)
    return os.path.abspath(os.path.join(base, f"{safe}__{branch_safe}.json"))


def load_repo_map(repo: str, branch: str) -> Optional[dict]:
    path = sidecar_path(repo, branch)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_repo_map(data: dict) -> str:
    path = sidecar_path(data["repo"], data["branch"])
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ── Heuristic extraction (no LLM, no AST) ───────────────────────────────────

def _classify(path: str) -> Optional[str]:
    name = os.path.basename(path)
    if name in _CONFIG_FILENAMES:
        return _SOURCE_EXTS.get(name.lower().split(".")[-1], "config")
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return _SOURCE_EXTS.get(ext)


def _should_skip_path(path: str) -> bool:
    parts = path.split("/")
    return any(p in _SKIP_DIRS for p in parts)


_SYMBOL_PATTERNS = {
    "python":     [r"^\s*(?:async\s+)?def\s+([a-zA-Z_][\w]*)", r"^\s*class\s+([A-Za-z_][\w]*)"],
    "java":       [r"^\s*(?:public|protected|private|static|final|abstract|synchronized|\s)+[<>\w\[\]\s]+?\s+([a-zA-Z_][\w]*)\s*\(",
                   r"^\s*(?:public\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+([A-Za-z_][\w]*)"],
    "kotlin":     [r"^\s*fun\s+([a-zA-Z_][\w]*)", r"^\s*class\s+([A-Za-z_][\w]*)", r"^\s*object\s+([A-Za-z_][\w]*)"],
    "javascript": [r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
                   r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)",
                   r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*="],
    "typescript": [r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
                   r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)",
                   r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)",
                   r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)"],
    "go":         [r"^\s*func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?([A-Za-z_][\w]*)",
                   r"^\s*type\s+([A-Za-z_][\w]*)"],
    "ruby":       [r"^\s*def\s+([A-Za-z_][\w]*)", r"^\s*class\s+([A-Za-z_][\w:]*)"],
    "rust":       [r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][\w]*)", r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][\w]*)"],
    "csharp":     [r"^\s*(?:public|internal|private|protected|static|async|\s)+[<>\w\[\]\s]+?\s+([A-Za-z_][\w]*)\s*\(",
                   r"^\s*(?:public\s+)?(?:sealed\s+|abstract\s+)?(?:class|interface|record|struct|enum)\s+([A-Za-z_][\w]*)"],
    "shell":      [r"^\s*(?:function\s+)?([a-zA-Z_][\w]*)\s*\(\s*\)"],
    "scala":      [r"^\s*def\s+([A-Za-z_][\w]*)", r"^\s*(?:case\s+)?class\s+([A-Za-z_][\w]*)"],
}


def _extract_symbols(content: str, lang: str, max_symbols: int = 12) -> list[str]:
    patterns = _SYMBOL_PATTERNS.get(lang, [])
    seen: list[str] = []
    for line in content.splitlines():
        if len(seen) >= max_symbols:
            break
        for pat in patterns:
            m = re.match(pat, line)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
                break
    return seen


_DOC_PREFIX = {
    "python": ('"""', "'''", "#"),
    "shell": ("#",),
    "sql": ("--",),
    "yaml": ("#",),
    "toml": ("#",),
    "ini": ("#",),
    "dockerfile": ("#",),
}
_C_COMMENT_START = ("//", "/*", "*", "/**")


def _extract_summary(content: str, lang: str) -> str:
    """First meaningful comment / docstring line."""
    for raw in content.splitlines()[:60]:
        line = raw.strip()
        if not line:
            continue
        # skip shebangs and directives
        if line.startswith("#!") or line.startswith("<?") or line.startswith("@"):
            continue
        if lang in _DOC_PREFIX and line.startswith(_DOC_PREFIX[lang]):
            cleaned = line.lstrip("#/*").lstrip('"\'').strip()
            if len(cleaned) > 5:
                return cleaned[:200]
        elif line.startswith(_C_COMMENT_START):
            cleaned = line.lstrip("/*").strip()
            if len(cleaned) > 5:
                return cleaned[:200]
    return ""


# ── GitHub API helpers (borrow platform detection from github_tools) ───────

def _headers_for(repo: str):
    from .github_tools import _headers_for as _h
    return _h(repo)


def _api_for(repo: str, path: str) -> str:
    from .github_tools import _api_for as _a
    return _a(repo, path)


def _clean_repo(repo: str) -> str:
    from .github_tools import _clean_repo as _c
    return _c(repo)


def _get_head_sha(repo: str, branch: str) -> str:
    clean = _clean_repo(repo)
    url = _api_for(repo, f"/repos/{clean}/commits/{branch}")
    r = requests.get(url, headers=_headers_for(repo), timeout=15)
    r.raise_for_status()
    return r.json()["sha"]


def _get_tree(repo: str, sha: str) -> list[dict]:
    clean = _clean_repo(repo)
    url = _api_for(repo, f"/repos/{clean}/git/trees/{sha}?recursive=1")
    r = requests.get(url, headers=_headers_for(repo), timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("truncated"):
        # Repo has > 100 000 files. Content is still usable but incomplete.
        pass
    return [n for n in body.get("tree", []) if n.get("type") == "blob"]


def _fetch_content(repo: str, path: str, ref: str) -> Optional[str]:
    clean = _clean_repo(repo)
    url = _api_for(repo, f"/repos/{clean}/contents/{path}?ref={ref}")
    try:
        r = requests.get(url, headers=_headers_for(repo), timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list):
            return None
        return base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
    except Exception:
        return None


# ── Main pipeline ───────────────────────────────────────────────────────────

def build_repo_map(repo: str, branch: str, force: bool = False, log=print) -> dict:
    """Build (or refresh) the sidecar for a repo+branch."""
    head_sha = _get_head_sha(repo, branch)

    existing = load_repo_map(repo, branch)
    if existing and not force and existing.get("head_sha") == head_sha:
        log(f"[repo_map] cache hit for {repo}@{branch} (sha={head_sha[:8]})")
        return existing

    log(f"[repo_map] building for {repo}@{branch} (sha={head_sha[:8]})")
    tree = _get_tree(repo, head_sha)
    log(f"[repo_map] tree returned {len(tree)} blobs")

    # Classify and filter
    candidates: list[dict] = []
    for node in tree:
        p = node["path"]
        if _should_skip_path(p):
            continue
        lang = _classify(p)
        if not lang:
            continue
        size = int(node.get("size", 0) or 0)
        if size > MAX_FILE_BYTES:
            continue
        candidates.append({"path": p, "lang": lang, "sha": node.get("sha", ""), "size": size})

    log(f"[repo_map] {len(candidates)} source/config files eligible")

    # Reuse content fetching for the first MAX_FILES_TO_FETCH sorted by size
    candidates.sort(key=lambda c: c["size"], reverse=True)
    to_fetch = candidates[:MAX_FILES_TO_FETCH]
    to_list_only = candidates[MAX_FILES_TO_FETCH:]

    fetched: dict[str, FileEntry] = {}

    def _worker(c: dict):
        content = _fetch_content(repo, c["path"], head_sha)
        if content is None:
            return c["path"], None
        entry = FileEntry(
            path=c["path"],
            lang=c["lang"],
            sha=c["sha"],
            size=c["size"],
            lines=content.count("\n") + 1,
            summary=_extract_summary(content, c["lang"]),
            symbols=_extract_symbols(content, c["lang"]),
        )
        return c["path"], entry

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = [pool.submit(_worker, c) for c in to_fetch]
        done = 0
        for f in as_completed(futures):
            try:
                path, entry = f.result()
                if entry:
                    with _lock:
                        fetched[path] = entry
                done += 1
                if done % 50 == 0:
                    log(f"[repo_map]   {done}/{len(to_fetch)} files summarised")
            except Exception as exc:
                log(f"[repo_map] worker error: {exc}")

    # Assemble final list
    files: list[dict] = []
    for c in to_fetch:
        entry = fetched.get(c["path"])
        if entry:
            files.append(asdict(entry))
        else:
            files.append({"path": c["path"], "lang": c["lang"], "sha": c["sha"],
                          "size": c["size"], "lines": 0, "summary": "", "symbols": []})
    for c in to_list_only:
        files.append({"path": c["path"], "lang": c["lang"], "sha": c["sha"],
                      "size": c["size"], "lines": 0, "summary": "",
                      "symbols": [], "note": "not summarised (over fetch cap)"})

    data = {
        "repo": _clean_repo(repo),
        "branch": branch,
        "head_sha": head_sha,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_files": len(files),
        "summarised_files": len(fetched),
        "files": files,
    }

    save_repo_map(data)
    log(f"[repo_map] saved sidecar → {sidecar_path(repo, branch)}")
    return data


def compact_view(data: dict, path_filter: Optional[str] = None, limit: int = 200) -> dict:
    """Return a slimmer view suitable for LLM context."""
    files = data.get("files", [])
    if path_filter:
        pf = path_filter.lower()
        files = [f for f in files if pf in f["path"].lower()]
    files = files[:limit]
    return {
        "repo": data.get("repo"),
        "branch": data.get("branch"),
        "head_sha": data.get("head_sha"),
        "showing": len(files),
        "total_files": data.get("total_files"),
        "files": [
            {
                "path": f["path"],
                "lang": f.get("lang"),
                "lines": f.get("lines", 0),
                "symbols": f.get("symbols", [])[:8],
                "summary": f.get("summary", ""),
            }
            for f in files
        ],
    }
