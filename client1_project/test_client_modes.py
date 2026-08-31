import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "client_module",
    Path(__file__).with_name("client1.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parse_train_and_connect_flags():
    args = module.parse_args(["--train", "--id", "client1", "--data", "data/client1/client1.yaml"])
    assert args.train is True
    assert args.connect is False

    connect_args = module.parse_args([
        "--connect",
        "--server",
        "127.0.0.1:8090",
        "--model",
        "results/client1_training/weights/best.pt",
    ])
    assert connect_args.connect is True
    assert connect_args.model == "results/client1_training/weights/best.pt"


def test_resolve_model_path_falls_back_to_training_outputs(tmp_path):
    results_dir = tmp_path / "results" / "client1_training" / "weights"
    results_dir.mkdir(parents=True)
    model_path = results_dir / "best.pt"
    model_path.write_bytes(b"model-state")

    resolved = module.resolve_model_path(
        project_root=tmp_path,
        run_name="client1_local",
        preferred_path=None,
    )

    assert resolved == model_path
