TARGET_HZ          = 64        # target sampling rate after resampling (Hz)
WINDOW_SEC         = 5         # window duration in seconds
MIN_OVERLAP        = 0.50      # minimum overlap for adaptive windowing
MAX_OVERLAP        = 0.85      # maximum overlap for adaptive windowing
TARGET_CLASS_RATIO = 0.85      # target minority-to-majority window ratio
LATENT_DIM         = 8         # encoder latent dimensionality
TARGET_DOMAIN      = "wearpd"  # reference domain for CORAL and MMD adaptation
RANDOM_STATE       = 42        # global random seed
