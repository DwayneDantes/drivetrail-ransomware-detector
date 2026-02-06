import argparse
import json
import os
import sqlite3
import time

import pandas as pd
import xgboost as xgb

from scripts.feature_extractor import extract_features

EVENT_DB_PATH = os.path.join('data', 'events.db')
MODEL_PATH = os.path.join('models', 'xgb_drivetrail.model')
DEFAULT_ALERT_LOG = os.path.join('data', 'baseline_xgb_alerts.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline detector: trigger when XGBoost event probability exceeds threshold."
    )
    parser.add_argument("--db", default=EVENT_DB_PATH, help="Path to events.db")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to XGBoost model")
    parser.add_argument("--threshold", type=float, default=0.9, help="Probability threshold")
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
    file_state_tracker = {}

    print("Baseline XGBoost Detector (event-level)")
    print(f"Polling every {args.poll_interval}s | threshold={args.threshold:.2f}")
    print(f"Logging alerts to {args.output}")

    model = xgb.XGBClassifier()
    model.load_model(args.model)

    try:
        while True:
            with sqlite3.connect(args.db) as conn:
                new_events_df = read_new_events(conn, last_processed_event_id)

            if not new_events_df.empty:
                for _, event in new_events_df.iterrows():
                    event_dict = event.to_dict()
                    feature_vector, file_state_tracker = extract_features(
                        event_dict, file_state_tracker
                    )
                    prob = model.predict_proba(feature_vector)[0][1]

                    if prob >= args.threshold:
                        alert_payload = {
                            "timestamp": time.time(),
                            "threshold": args.threshold,
                            "probability": float(prob),
                            "event_id": int(event_dict["id"]),
                            "event_type": event_dict.get("event_type"),
                            "local_title": event_dict.get("local_title"),
                            "full_path": event_dict.get("full_path"),
                        }
                        print(
                            f"[ALERT] p={prob:.3f} event_id={event_dict['id']} "
                            f"type={event_dict.get('event_type')}"
                        )
                        append_alert(args.output, alert_payload)

                last_processed_event_id = int(new_events_df["id"].max())

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nBaseline XGBoost Detector stopped.")


if __name__ == "__main__":
    main()
