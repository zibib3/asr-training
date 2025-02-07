#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to engine script (e.g. engines/faster_whisper_engine.py)")
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--output-dir", required=True, help="Directory to store evaluation results")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    # Ensure engine script exists
    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine script not found: {args.engine}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define dataset configurations as list of tuples
    datasets = [
        ("ivrit-ai/eval-d1:test:text", None, "ivrit_ai_eval_d1"),
        ("ivrit-ai/saspeech:test:text", None, "saspeech"),
        ("google/fleurs:test:transcription", "he_il", "fleurs"),
        ("mozilla-foundation/common_voice_17_0:test:sentence", "he", "common_voice_17"),
        ("imvladikon/hebrew_speech_kan:validation:sentence", None, "hebrew_speech_kan"),
    ]

    # Iterate over datasets and run evaluation
    for ds_path, ds_name, ds_output_name in datasets:
        output_file = os.path.join(args.output_dir, f"{ds_output_name}.csv")

        reevaluate_existing = False
        if os.path.exists(output_file):
            if not args.overwrite:
                reevaluate_existing = True
                print(f"Reevaluating exiting results in {ds_path}.")

        print(f"Evaluating {ds_path}...")

        cmd = [
            "./evaluate_model.py",
            "--engine",
            str(engine_path.absolute()),
            "--model",
            args.model,
            "--dataset",
            ds_path,
            "--workers",
            str(args.workers),
            "--output",
            output_file,
        ]

        if reevaluate_existing:
            cmd.append("--reevaluate_existing")
        if ds_name:
            cmd.extend(["--name", ds_name])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {ds_path}: {e}")
            continue


if __name__ == "__main__":
    main()
