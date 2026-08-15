import logging
import os
from pathlib import Path

import click
from logging_config import setup_logging

PID_FILE = Path("/tmp/wms-sop-mcp.pid")

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """WMS MCP CLI."""

@cli.command()
def start():
    """Start the WMS SOP MCP server."""
    from app import mcp, settings

    setup_logging(log_level=settings.log_level, log_file=settings.log_file)
    logger.info("Starting %s on port %d (log_file=%s)", settings.service_name, settings.port, settings.log_file)

    PID_FILE.write_text(str(os.getpid()))
    try:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.port)
    finally:
        PID_FILE.unlink(missing_ok=True)