#!/usr/bin/env python3
"""Send a summarized Telegram notification for pipeline runs."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional, Set

import urllib.request
from zoneinfo import ZoneInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send Telegram summary message")
    parser.add_argument("--log-path", default="pipeline.log", help="Path to pipeline log file")
    parser.add_argument("--timezone", default="America/Bogota", help="IANA timezone for timestamp")
    parser.add_argument("--title", default="🚀 Pipeline publicacion", help="Message title")
    parser.add_argument("--status", default="unknown", help="Pipeline status (success/failure)")
    return parser.parse_args()


RUN_COMPLETE_RE = re.compile(r"Run complete:\s*(?P<optimized>\d+) posts optimized,\s*(?P<skipped>\d+) skipped")


def build_summary(log_path: str) -> str:
    if not os.path.isfile(log_path):
        return "Resumen no disponible"
    run_complete: Optional[str] = None
    skipped_items: Optional[str] = None
    optimized_count = 0
    skipped_count = 0
    errors: Set[str] = set()
    failure_reasons: List[str] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "Run complete:" in line:
                run_complete = line.strip()
                match = RUN_COMPLETE_RE.search(line)
                if match:
                    optimized_count = max(optimized_count, int(match.group("optimized")))
                    skipped_count = max(skipped_count, int(match.group("skipped")))
            if "Skipped items:" in line:
                skipped_items = line.strip()
            if "Optimized post" in line:
                optimized_count += 1
            if "Skipping post" in line:
                skipped_count += 1
                failure_reasons.append(line.strip())
            if "ERROR" in line or "Failed" in line or "Traceback" in line:
                errors.add(line.strip())
    # Si no hubo errores explícitos pero se registraron Skipping posts, muéstralos
    if failure_reasons and not errors:
        errors.update(failure_reasons)
    summary_parts = []
    summary_parts.append(f"Posts optimizados: {optimized_count}")
    summary_parts.append(f"Posts omitidos: {skipped_count}")
    if run_complete:
        summary_parts.append(run_complete)
    if skipped_items and skipped_items not in errors:
        summary_parts.append(skipped_items)
    if errors:
        truncated_errors = []
        for item in sorted(errors)[:3]:
            cleaned = item.strip()
            if len(cleaned) > 220:
                cleaned = cleaned[:217] + "..."
            truncated_errors.append(cleaned)
        summary_parts.append("Errores detectados:\n- " + "\n- ".join(truncated_errors))
    return "\n".join(summary_parts) if summary_parts else "Resumen no disponible"


def send_message(token: str, chat_id: str, text: str) -> None:
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:  # nosec B310
        if response.status != 200:
            raise RuntimeError(f"Telegram notification failed: {response.status}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    summary = build_summary(args.log_path)
    tz = ZoneInfo(args.timezone)
    now = datetime.now(timezone.utc).astimezone(tz)
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    message = (
        f"{args.title}\n"
        f"Estado: {args.status}\n"
        f"Fecha ({args.timezone}): {formatted_time}\n"
        f"{summary}"
    )
    send_message(token, chat_id, message)


if __name__ == "__main__":
    main()
