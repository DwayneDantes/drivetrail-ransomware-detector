import argparse
import json
import os
import sqlite3
import time

import pandas as pd

EVENT_DB_PATH = os.path.join('data', 'events.db')
DEFAULT_ALERT_LOG = os.path.join('data', 'baseline_threshold_alerts.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline detector: trigger if modify/rename count exceeds threshold."
    )
    parser.add_argument("--db", default=EVENT_DB_PATH, help="Path to events.db")
    parser.add_argument("--window-seconds", type=int, default=30, help="Window size in seconds")
    parser.add_argument("--threshold", type=int, default=15, help="Event count threshold")
    parser.add_argument("--poll-interval", type=int, default=3, help="Polling interval in seconds")
    parser.add_argument("--output", default=DEFAULT_ALERT_LOG, help="Alert log output path")
    return parser.parse_args()


def read_new_events(conn, last_processed_id):
    query = f"SELECT * FROM events WHERE id > {last_processed_id} ORDER BY id ASC"
    return pd.read_sql_query(query, conn)


def append_alert(log_path, alert_payload):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = []
    data.append(alert_payload)
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main():
    args = parse_args()
    last_processed_event_id = 0
    recent_events_df = pd.DataFrame()

    print("Baseline Threshold Detector (MODIFIED/RENAMED count)")
    print(
        f"Polling every {args.poll_interval}s | window={args.window_seconds}s | "
        f"threshold={args.threshold}"
    )
    print(f"Logging alerts to {args.output}")

    try:
        while True:
            with sqlite3.connect(args.db) as conn:
                new_events_df = read_new_events(conn, last_processed_event_id)

            if not new_events_df.empty:
                recent_events_df = pd.concat([recent_events_df, new_events_df], ignore_index=True)
                last_processed_event_id = int(new_events_df["id"].max())

                current_time = time.time()
                window_start = current_time - args.window_seconds
                recent_events_df = recent_events_df[recent_events_df["timestamp"] >= window_start]

                filtered = recent_events_df[
                    recent_events_df["event_type"].isin(["MODIFIED", "RENAMED"])
                ]
                event_count = len(filtered)

                if event_count >= args.threshold:
                    alert_payload = {
                        "timestamp": current_time,
                        "window_seconds": args.window_seconds,
                        "threshold": args.threshold,
                        "event_count": event_count,
                        "last_event_id": last_processed_event_id,
                    }
                    print(
                        f"[ALERT] {event_count} MODIFY/RENAME events in "
                        f"{args.window_seconds}s window."
                    )
                    append_alert(args.output, alert_payload)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nBaseline Threshold Detector stopped.")


if __name__ == "__main__":
    main()
