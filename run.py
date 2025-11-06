
import argparse, os, pandas as pd, numpy as np
from common.io import load_yaml, load_csv_or_none, ensure_dir, save_json
from common import synth
from pathlib import Path

def load_project_config(pid):
    cfg_path = Path("projects") / pid / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Unknown project id: {pid}")
    return load_yaml(cfg_path)

def make_data(task):
    if task=="tabular_classification":
        return synth.make_classification_df(n=1000, p=12)
    if task=="tabular_regression":
        return synth.make_regression_df(n=1000, p=12)
    if task=="time_series_forecast":
        return synth.make_ts_df(n=220)
    if task=="nlp_text_classification":
        return synth.make_sentiment_df(n=800)
    if task=="recommendation_cf":
        return synth.make_reco_df()
    if task=="anomaly_detection":
        return synth.make_regression_df(n=600, p=6).drop(columns=["target"])
    if task=="survival_analysis":
        df = synth.make_classification_df(n=600, p=6)
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        df["duration"] = (rng.exponential(10, size=len(df))* (1+df["f0"].abs())).astype(float)
        df["event"] = (rng.random(len(df))>0.3).astype(int)
        return df.drop(columns=["target"])
    return None

def run_project(pid, data_path=None, target=None):
    cfg = load_project_config(pid)
    task = cfg["task"]
    out_dir = ensure_dir(os.path.join("outputs", pid))
    if task=="graph_route_optimization":
        from tasks import graph_route_optimization as eng
        n_nodes = cfg.get("n_nodes", 15)
        res = eng.run(n_nodes=n_nodes, out_dir=out_dir)
        save_json(res, os.path.join(out_dir,"result.json"))
        print("Route optimization done. See outputs folder.")
        return

    # tabular-like
    df = load_csv_or_none(data_path) or make_data(task)
    if df is None:
        raise SystemExit("No data available and synthesizer missing for this task.")
    if task in ("tabular_classification","tabular_regression","nlp_text_classification","time_series_forecast","recommendation_cf","anomaly_detection","survival_analysis"):
        if task=="nlp_text_classification":
            tgt = target or cfg.get("target","target")
            from tasks import nlp_text_classification as eng
            rep = eng.run(df, tgt, out_dir)
            print("NLP text classification complete.")
        elif task=="time_series_forecast":
            from tasks import time_series_forecast as eng
            res = eng.run(df, target or cfg.get("target","y"), out_dir)
            save_json(res, os.path.join(out_dir,"forecast_meta.json"))
            print("Time-series forecasting complete.")
        elif task=="recommendation_cf":
            from tasks import recommendation_cf as eng
            res = eng.run(df, out_dir)
            print("Recommendation training complete.")
        elif task=="anomaly_detection":
            from tasks import anomaly_detection as eng
            res = eng.run(df.select_dtypes(include=[float,int]), out_dir)
            save_json(res, os.path.join(out_dir,"anomaly_meta.json"))
            print("Anomaly detection complete.")
        elif task=="survival_analysis":
            from tasks import survival as eng
            res = eng.run(df, out_dir)
            save_json(res, os.path.join(out_dir,"survival_meta.json"))
            print("Survival analysis complete.")
        else:
            tgt = target or cfg.get("target","target")
            if task=="tabular_classification":
                from tasks import tabular_classification as eng
            else:
                from tasks import tabular_regression as eng
            rep = eng.run(df, tgt, out_dir)
            print("Training complete. Metrics saved.")
    else:
        raise SystemExit(f"Unsupported task type: {task}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="Project id (e.g., 01_disease_prediction)")
    ap.add_argument("--data", help="Path to CSV (optional)")
    ap.add_argument("--target", help="Target column if overriding config")
    ap.add_argument("--list", action="store_true", help="List projects")
    args = ap.parse_args()

    if args.list:
        for p in sorted(os.listdir("projects")):
            if os.path.isdir(os.path.join("projects", p)):
                print(p)
        return

    if not args.project:
        ap.error("--project is required unless --list")

    run_project(args.project, data_path=args.data, target=args.target)

if __name__ == "__main__":
    main()
