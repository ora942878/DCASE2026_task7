from pathlib import Path
"""
from configs.CFG_PATH import PATH
from configs.CFG_PATH import CFG
"""

class CFG:
    duration = 4
    sample_rate = 32000
    clip_samples = sample_rate * duration
    seed = 3407

    NUM_TASKS = 3
    NUM_WORKERS = 0
    PIN_MEMORY = True

    mel_bins = 64
    fmin = 50
    fmax = 14000
    window_size = 1024
    hop_size = 320
    window = 'hann'
    pad_mode = 'reflect'
    center = True
    device = 'cuda'
    ref = 1.0
    amin = 1e-10
    top_db = None
    classes_num_DIL = 10

    # ==== Loader parameters ====
    batch_size = 32
    epochs = 20
    learning_rate = 1e-4 # LR_D2\D3 = learning_rate * 0.1 in official baseline

    dict_class_labels = { 'alarm': 0,
                   'baby_cry': 1,
                   'dog_bark': 2,
                   'engine': 3,
                   'fire': 4,
                   'footsteps': 5,
                   'knocking': 6,
                   'telephone_ringing': 7,
                   'piano': 8,
                   'speech': 9
                   }

class PATH:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CODE_ROOT = PROJECT_ROOT.parent
    REPO_ROOT = CODE_ROOT.parent

    # ======= DATA =======
    DATA_ROOT   = REPO_ROOT / "data"

    RAW_DATA_D2     = DATA_ROOT     / "D2"
    RAW_DATA_D3     = DATA_ROOT     / "D3"

    RAW_DATA_EVAL   = DATA_ROOT     / "eval"
    TEST_D2         = RAW_DATA_D2       / "d2-dev-test"
    TEST_D2_CSV     = RAW_DATA_D2       / "metadata" / "d2-dev-test.csv"
    TRAIN_D2        = RAW_DATA_D2       / "d2-dev-train"
    TRAIN_D2_CSV    = RAW_DATA_D2       / "metadata" / "d2-dev-train.csv"
    TEST_D3         = RAW_DATA_D3       / "d3-dev-test"
    TEST_D3_CSV     = RAW_DATA_D3       / "metadata" / "d3-dev-test.csv"
    TRAIN_D3        = RAW_DATA_D3       / "d3-dev-train"
    TRAIN_D3_CSV    = RAW_DATA_D3       / "metadata" / "d3-dev-train.csv"

    # ======= PROCESSED_DATA =======
    PROCESSED_DATA      = DATA_ROOT         / "processed_data"
    TRAIN_D2_CHUNK_4    = PROCESSED_DATA    / "D2-train-chunk-4"
    TEST_D2_CHUNK_4     = PROCESSED_DATA    / "D2-test-chunk-4"
    TRAIN_D3_CHUNK_4    = PROCESSED_DATA    / "D3-train-chunk-4"
    TEST_D3_CHUNK_4     = PROCESSED_DATA    / "D3-test-chunk-4"

    small_sample_set    = PROCESSED_DATA    / "small_sample_set"
    # ======= META_DATA =======
    METADATA_ROOT   = PROJECT_ROOT  / "metadata"

    # ======= CHECKPOINT & RUNS =======
    CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"

    OFFICIAL_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "official"
    OFFICIAL_PROCESSED_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "official"
    MY_CHECKPOINT_ROOT01 = CHECKPOINT_ROOT / "ours"

    EXP_CHECKPOINT_ROOT01 = CHECKPOINT_ROOT / "EXP_01"

    RUNS = PROJECT_ROOT / "runs"


if __name__ == "__main__":
    from pathlib import Path

    print("Checking all paths in PATH")

    path_items = []

    for name in dir(PATH):
        if name.startswith("__"):
            continue

        value = getattr(PATH, name)

        if isinstance(value, Path):
            path_items.append((name, value))

    path_items = sorted(path_items, key=lambda x: x[0])

    for name, path in path_items:
        exists = path.exists()

        if exists:
            if path.is_dir():
                path_type = "DIR "
            elif path.is_file():
                path_type = "FILE"
            else:
                path_type = "OTHER"
            status = "OK"
        else:
            path_type = "----"
            status = "MISSING"

        print(f"{status:8s} | {path_type:5s} | {name:25s} | {path}")

    print("=" * 80)

    missing_paths = [
        (name, path)
        for name, path in path_items
        if not path.exists()
    ]

    if missing_paths:
        print("Missing paths:")
        for name, path in missing_paths:
            print(f"  - {name}: {path}")
    else:
        print("All paths exist.")
