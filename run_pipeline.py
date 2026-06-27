import argparse
import time
import tensorflow  
def run_step(label, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full postural instability pipeline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print(f"\nPostural Instability Pipeline")
    print(f"  seed : {args.seed}")

    # Step 2 — Preprocessing
    import preprocessing
    run_step("Step 2 — Preprocessing", preprocessing.main)

    # Step 3 — Feature extraction (both arch modes)
    import feature_extraction
    for arch in ["classifier", "autoencoder"]:
        run_step(
            f"Step 3 — Feature extraction ({arch})",
            feature_extraction.main,
            arch_mode=arch,
            seed=args.seed,
        )

    # Step 4a — Classification (both arch modes)
    import classifier
    for arch in ["classifier", "autoencoder"]:
        run_step(
            f"Step 4a — Postural instability classification ({arch})",
            classifier.run_ablation,
            arch_mode=arch,
            seed=args.seed,
        )

    # Step 4b — Domain separability (both arch modes)
    import domain_classifier
    run_step(
        "Step 4b — Domain separability analysis",
        domain_classifier.main,
        arch_mode_filter=None,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"  Results saved to: results/")
    print(f"  Models saved to:  models/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
