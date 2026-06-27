import argparse
import sys
import time


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
        description="Run the full postural instability pipeline (steps 2-4)."
    )
    parser.add_argument(
        "--arch-mode",
        choices=["autoencoder", "classifier"],
        default="autoencoder",
        help="Encoder architecture to use (default: autoencoder)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip step 2 if preprocessed data already exists",
    )
    args = parser.parse_args()

    print(f"\nPostural Instability Pipeline")
    print(f"  arch-mode : {args.arch_mode}")
    print(f"  seed      : {args.seed}")

    # Step 2 — Preprocessing
    if not args.skip_preprocessing:
        import preprocessing
        run_step("Step 2 — Preprocessing", preprocessing.main)
    else:
        print("\nStep 2 skipped (--skip-preprocessing).")

    # Step 3 — Feature extraction
    import feature_extraction
    run_step(
        f"Step 3 — Feature extraction ({args.arch_mode})",
        feature_extraction.main,
        arch_mode=args.arch_mode,
        seed=args.seed,
    )

    # Step 4a — Classification
    import classifier
    run_step(
        f"Step 4a — Postural instability classification ({args.arch_mode})",
        classifier.run_ablation,
        arch_mode=args.arch_mode,
        seed=args.seed,
    )

    # Step 4b — Domain separability
    import domain_classifier
    run_step(
        f"Step 4b — Domain separability analysis ({args.arch_mode})",
        domain_classifier.main,
        arch_mode_filter=args.arch_mode,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"  Results saved to: posturalInstability/results/")
    print(f"  Models saved to:  posturalInstability/models/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
