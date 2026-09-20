"""MCP server entry point. Speaks over stdio so a client can launch it as a subprocess."""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from opentargets_mcp import tools
from opentargets_mcp.constants import (
    DEFAULT_ASSOCIATION_ROWS,
    DEFAULT_DRUG_ROWS,
    DEFAULT_EVIDENCE_ROWS,
)
from opentargets_mcp.transport import OpenTargetsError

logger = logging.getLogger(__name__)

mcp = MCPServer("opentargets", version="0.1.0")


@mcp.tool()
async def resolve(text: str, kind: str | None = None) -> str:
    """Look up what Open Targets identifier a name maps to.

    Args:
        text: A gene symbol, disease name, or drug name.
        kind: One of target, disease, drug. Omit to search all three.
    """
    return await _guard(tools.tool_resolve(text, kind))


@mcp.tool()
async def target_profile(gene: str) -> str:
    """Identity, function, tractability, safety liabilities, and genetic constraint for a target.

    Args:
        gene: A gene symbol such as JAK1.
    """
    return await _guard(tools.tool_target_profile(gene))


@mcp.tool()
async def target_diseases(gene: str, count: int = DEFAULT_ASSOCIATION_ROWS) -> str:
    """Diseases most strongly associated with a target, by overall association score.

    Args:
        gene: A gene symbol such as JAK1.
        count: How many rows to return.
    """
    return await _guard(tools.tool_target_diseases(gene, count))


@mcp.tool()
async def disease_targets(disease: str, count: int = DEFAULT_ASSOCIATION_ROWS) -> str:
    """Targets most strongly associated with a disease, by overall association score.

    Args:
        disease: A disease name such as psoriasis.
        count: How many rows to return.
    """
    return await _guard(tools.tool_disease_targets(disease, count))


@mcp.tool()
async def known_drugs(name: str, kind: str = "target", count: int = DEFAULT_DRUG_ROWS) -> str:
    """Drugs and clinical candidates recorded against a target or a disease.

    Args:
        name: A gene symbol or a disease name.
        kind: Whether `name` is a target or a disease.
        count: How many rows to return.
    """
    if kind not in ("target", "disease"):
        return "Error: kind must be either target or disease."
    return await _guard(tools.tool_known_drugs(name, kind, count))


@mcp.tool()
async def evidence(gene: str, disease: str, count: int = DEFAULT_EVIDENCE_ROWS) -> str:
    """The evidence rows Open Targets holds for one target-disease pair.

    Args:
        gene: A gene symbol such as JAK1.
        disease: A disease name such as psoriasis.
        count: How many rows to return.
    """
    return await _guard(tools.tool_evidence(gene, disease, count))


async def _guard(coro) -> str:
    """Turn a lookup failure into a readable message rather than a traceback in the client."""
    try:
        return await coro
    except OpenTargetsError as exc:
        logger.warning("Open Targets lookup failed: %s", exc)
        return f"Error: {exc}"


def main() -> None:
    """Run the server over stdio."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
