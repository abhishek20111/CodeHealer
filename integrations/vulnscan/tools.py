"""VulnScan LangChain tools for repository insights, vulnerabilities, and secrets."""
import json
from langchain_core.tools import tool
from integrations.vulnscan.query_client import query_graphql, DEFAULT_APP_ID


SEVERITY_MAP = {4: "critical", 3: "high", 2: "medium", 1: "low", 0: "info"}
MAX_VulnScan_LIMIT = 50  # VulnScan API max is 100, keep below


def create_vulnscan_tools():
    """Create and return all VulnScan tools."""

    @tool
    def list_scan_repositories(app_id: str = DEFAULT_APP_ID) -> str:
        """List all repositories registered in VulnScan for a given APP_ID (asset ID).
        Returns repo names, visibility, languages, and archive status.

        Args:
            app_id: The Central ID (asset ID) to query. Defaults to configured APP_ID.
        """
        data = query_graphql("""
        {
            repositoryInsights(condition: { centralid: "%s" }, first: 50) {
                nodes { name visibility languages }
                totalCount
            }
        }
        """ % app_id)
        repos = data["repositoryInsights"]
        results = []
        for r in repos["nodes"]:
            langs = ", ".join(l["name"] for l in (r["languages"] or []))
            results.append(f"- {r['name']} ({r['visibility']}) [{langs}]")
        return f"Total: {repos['totalCount']} repositories\n\n" + "\n".join(results)

    @tool
    def get_scan_vulnerabilities(app_id: str = DEFAULT_APP_ID, severity: str = "all", limit: int = 20) -> str:
        """Get vulnerable dependencies from VulnScan for a APP_ID.
        Shows package name, version, CVEs, severity, safe version, and which repo/file is affected.

        Args:
            app_id: The Central ID (asset ID). Defaults to configured APP_ID.
            severity: Filter by severity: "critical", "high", "medium", "low", or "all". Default "all".
            limit: Max results to return (default 20, max 50).
        """
        limit = min(limit, MAX_VulnScan_LIMIT)
        severity_filter = ""
        if severity and severity != "all":
            sev_num = {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(severity.lower())
            if sev_num:
                severity_filter = f', severityGroup: {{ equalTo: {sev_num} }}'

        data = query_graphql("""
        {
            dependencyInsights(
                filter: { centralid: { equalTo: "%s" }, vulnerabilities: { isNull: false }%s }
                first: %d
                orderBy: MAX_SEVERITY_SCORE_DESC
            ) {
                nodes {
                    name version displaySeverity maxSeverityScore
                    vulnerabilities allSafeVersions repository fileName
                }
                totalCount
            }
        }
        """ % (app_id, severity_filter, limit))

        deps = data["dependencyInsights"]
        if deps["totalCount"] == 0:
            return f"No vulnerable dependencies found for APP_ID {app_id}" + (f" with severity={severity}" if severity != "all" else "")

        results = [f"**{deps['totalCount']} vulnerable packages** (showing top {min(limit, len(deps['nodes']))})\n"]
        for d in deps["nodes"]:
            cves = ", ".join(d["vulnerabilities"][:5])
            if len(d["vulnerabilities"]) > 5:
                cves += f" (+{len(d['vulnerabilities'])-5} more)"
            safe = d["allSafeVersions"][-1] if d.get("allSafeVersions") else "unknown"
            results.append(
                f"| {d['displaySeverity'].upper():8s} | {d['name']}@{d['version']} | "
                f"Score: {d['maxSeverityScore']} | Safe: {safe} | "
                f"Repo: {d['repository']} | File: {d['fileName']}"
            )
        return "\n".join(results)

    @tool
    def get_scan_secrets(app_id: str = DEFAULT_APP_ID, limit: int = 20) -> str:
        """Get sensitive data elements (secrets) detected by Earlybird for a APP_ID.
        Shows secret category and file location.

        Args:
            app_id: The Central ID (asset ID). Defaults to configured APP_ID.
            limit: Max results to return (default 20, max 50).
        """
        limit = min(limit, MAX_VulnScan_LIMIT)
        data = query_graphql("""
        {
            sensitiveDataElementInsights(
                condition: { centralid: "%s" }
                first: %d
            ) {
                nodes { category fileName }
                totalCount
            }
        }
        """ % (app_id, limit))

        secrets = data["sensitiveDataElementInsights"]
        if secrets["totalCount"] == 0:
            return f"No secrets detected for APP_ID {app_id}"

        results = [f"**{secrets['totalCount']} secrets detected** (showing {min(limit, len(secrets['nodes']))})\n"]
        for s in secrets["nodes"]:
            results.append(f"- [{s['category']}] {s['fileName']}")
        return "\n".join(results)

    @tool
    def search_scan_dependency(package_name: str, app_id: str = DEFAULT_APP_ID) -> str:
        """Search for a specific package across all repos in a APP_ID.
        Useful to find which repos use a particular dependency and their versions.

        Args:
            package_name: Name of the package to search (e.g. "log4j", "axios", "spring-core").
            app_id: The Central ID. Defaults to configured APP_ID.
        """
        data = query_graphql("""
        {
            dependencyInsights(
                filter: {
                    centralid: { equalTo: "%s" }
                    name: { includesInsensitive: "%s" }
                }
                first: 30
                orderBy: MAX_SEVERITY_SCORE_DESC
            ) {
                nodes {
                    name version displaySeverity maxSeverityScore
                    vulnerabilities allSafeVersions repository fileName
                }
                totalCount
            }
        }
        """ % (app_id, package_name))

        deps = data["dependencyInsights"]
        if deps["totalCount"] == 0:
            return f"Package '{package_name}' not found in APP_ID {app_id}"

        results = [f"**{deps['totalCount']} entries** for '{package_name}' across repos\n"]
        for d in deps["nodes"]:
            vuln_count = len(d["vulnerabilities"]) if d["vulnerabilities"] else 0
            safe = d["allSafeVersions"][-1] if d.get("allSafeVersions") else "-"
            sev = d["displaySeverity"] or "none"
            results.append(
                f"- {d['repository']}/{d['fileName']}: {d['name']}@{d['version']} "
                f"| Severity: {sev} | CVEs: {vuln_count} | Safe: {safe}"
            )
        return "\n".join(results)

    @tool
    def get_scan_repo_dependencies(repo_name: str, app_id: str = DEFAULT_APP_ID, vulnerable_only: bool = True) -> str:
        """Get all dependencies for a specific repository.
        Can filter to show only vulnerable ones or all.

        Args:
            repo_name: Repository name (e.g. "sampleapp-ui" or partial match).
            app_id: The Central ID. Defaults to configured APP_ID.
            vulnerable_only: If True, only show dependencies with vulnerabilities. Default True.
        """
        vuln_filter = ", vulnerabilities: { isNull: false }" if vulnerable_only else ""
        data = query_graphql("""
        {
            dependencyInsights(
                filter: {
                    centralid: { equalTo: "%s" }
                    repository: { includesInsensitive: "%s" }
                    %s
                }
                first: 50
                orderBy: MAX_SEVERITY_SCORE_DESC
            ) {
                nodes {
                    name version displaySeverity maxSeverityScore
                    vulnerabilities allSafeVersions fileName type
                }
                totalCount
            }
        }
        """ % (app_id, repo_name, vuln_filter))

        deps = data["dependencyInsights"]
        label = "vulnerable " if vulnerable_only else ""
        if deps["totalCount"] == 0:
            return f"No {label}dependencies found for repo '{repo_name}' in APP_ID {app_id}"

        results = [f"**{deps['totalCount']} {label}dependencies** in '{repo_name}'\n"]
        for d in deps["nodes"]:
            vuln_count = len(d["vulnerabilities"]) if d["vulnerabilities"] else 0
            safe = d["allSafeVersions"][-1] if d.get("allSafeVersions") else "-"
            sev = (d["displaySeverity"] or "none").upper()
            results.append(
                f"| {sev:8s} | {d['name']}@{d['version']} → {safe} | "
                f"CVEs: {vuln_count} | File: {d['fileName']}"
            )
        return "\n".join(results)

    return [
        list_scan_repositories,
        get_scan_vulnerabilities,
        get_scan_secrets,
        search_scan_dependency,
        get_scan_repo_dependencies,
    ]
