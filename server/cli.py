import logging
import os
from pathlib import Path

import click
from logging_config import setup_logging

PID_FILE = Path("/tmp/wms-sop-mcp.pid")

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
def start():
    from prometheus_client import start_http_server
    from telemetry import setup_telemetry

    setup_telemetry("wms-sop-mcp")
    from app import mcp, settings

    start_http_server(settings.metrics_port)
    setup_logging(log_level=settings.log_level, log_file=settings.log_file)
    logger.info(
        "Starting %s on port %d (log_file=%s)",
        settings.service_name,
        settings.port,
        settings.log_file,
    )

    PID_FILE.write_text(str(os.getpid()))
    try:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.port)
    finally:
        PID_FILE.unlink(missing_ok=True)
