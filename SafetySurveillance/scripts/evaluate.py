"""
QASD Automated Benchmark & Evaluation Runner
Executes Experiments 1 to 5 to evaluate mAP@50, Precision, Recall, F1,
Incident Accuracy, Anomaly AUC, and Real-Time FPS.
Generates publication-ready ASCII tables and JSON summaries.
"""

import json
import time
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from SafetySurveillance.evaluation.benchmarks import ResearchBenchmarkSuite


def main():
    print("=" * 80)
    print("  QASD - RESEARCH EXPERIMENT BENCHMARK EVALUATOR")
    print("  Quality-Aware Deep Learning Framework in Degraded CCTV Environments")
    print("=" * 80)
    print("\nRunning Experiments 1 through 5 across CCTV condition test sets...\n")

    time.sleep(1.0)
    results = ResearchBenchmarkSuite.run_full_suite()

    header = f"{'Experiment':<35} | {'mAP@50':<8} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Inc.Acc':<8} | {'FPS':<6}"
    print(header)
    print("-" * len(header))

    for exp in results["experiments"]:
        row = f"{exp['name']:<35} | {exp['map50']:<8.3f} | {exp['precision']:<7.3f} | {exp['recall']:<7.3f} | {exp['f1_score']:<7.3f} | {exp['incident_accuracy']:<8.3f} | {exp['fps']:<6.1f}"
        print(row)

    print("-" * len(header))
    print("\nKey Empirical Findings:")
    for k, v in results["summary"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    # Save to disk
    out_file = BASE_DIR / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Evaluation] Detailed report exported to: {out_file.name}")


if __name__ == "__main__":
    main()
