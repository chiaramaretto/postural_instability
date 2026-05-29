import feature_extraction as fe


LAMBDA_VALUES = [0.001, 0.005, 0.01]


def run_sweep():
    original_arch_mode = fe.ARCH_MODE
    original_use_mmd = fe.USE_MMD
    original_lambda = fe.LAMBDA_MMD

    try:
        fe.ARCH_MODE = "classifier"
        fe.USE_MMD = True

        for lambda_mmd in LAMBDA_VALUES:
            fe.LAMBDA_MMD = lambda_mmd
            print("\n" + "═" * 72)
            print(f"Running conditional MMD classifier sweep with lambda={lambda_mmd:.3f}")
            print("═" * 72)
            fe.main()
    finally:
        fe.ARCH_MODE = original_arch_mode
        fe.USE_MMD = original_use_mmd
        fe.LAMBDA_MMD = original_lambda


if __name__ == "__main__":
    run_sweep()