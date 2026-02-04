"""
Test script for SASLM training infrastructure.

Run this to verify all components work before starting actual training.

Usage:
    python src/training/test_infrastructure.py

Expected output:
    - All tests pass
    - Sample checkpoint saved/loaded
    - Grokking detection simulated
    - Config loading verified
"""

import sys
import os
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np


def test_checkpoint_manager():
    """Test checkpoint save/load functionality."""
    print("\n" + "="*60)
    print("Testing CheckpointManager...")
    print("="*60)

    from src.training.checkpoint_manager import CheckpointManager

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Initialize manager with local paths (not Colab)
        manager = CheckpointManager(
            experiment_name='test_experiment',
            base_path=temp_dir,
            local_cache=temp_dir,
            save_every_n_steps=10,
        )

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Save checkpoint
        metrics = {'train_loss': 0.5, 'val_loss': 0.6}
        manager.save(model, optimizer, step=100, metrics=metrics)
        print("  [PASS] Checkpoint saved")

        # Save best model
        manager.save_best(model, step=100, val_loss=0.6)
        print("  [PASS] Best model saved")

        # Create new model and load
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

        start_step, loaded_metrics = manager.load(model2, optimizer2, device='cpu')

        assert start_step == 100, f"Expected step 100, got {start_step}"
        assert loaded_metrics['train_loss'] == 0.5
        print("  [PASS] Checkpoint loaded")

        # Verify model weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Model weights don't match!"
        print("  [PASS] Model weights verified")

        # Test samples saving
        manager.save_samples(
            step=100,
            samples=["The Divine is...", "Supermind manifests..."],
            prompts=["The Divine", "Supermind"]
        )
        print("  [PASS] Samples saved")

        # Test metrics history
        history = manager.get_metrics_history()
        assert len(history) > 0
        print("  [PASS] Metrics history retrieved")

        print("\n  CheckpointManager: ALL TESTS PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_grokking_detector():
    """Test grokking detection."""
    print("\n" + "="*60)
    print("Testing GrokkingDetector...")
    print("="*60)

    from src.training.grokking_detector import GrokkingDetector

    # Create detector with smaller windows for testing
    detector = GrokkingDetector(
        window_size=50,
        min_plateau_steps=100,
        drop_threshold=0.15,
    )

    np.random.seed(42)

    # Simulate training phases
    grok_detected = False

    # Phase 1: Train loss drops, val stays high
    print("  Simulating memorization phase...")
    for step in range(0, 500, 10):
        train = 2.0 * np.exp(-step / 100) + 0.1
        val = 2.0 + np.random.normal(0, 0.02)
        detector.update(step, train, val)

    # Phase 2: Plateau
    print("  Simulating plateau phase...")
    for step in range(500, 1500, 10):
        train = 0.15 + np.random.normal(0, 0.01)
        val = 2.0 + np.random.normal(0, 0.02)
        detector.update(step, train, val)

    # Phase 3: Grokking
    print("  Simulating grokking phase...")
    for step in range(1500, 2500, 10):
        progress = min((step - 1500) / 300, 1.0)
        train = 0.15 + np.random.normal(0, 0.01)
        val = 2.0 - progress * 1.7 + np.random.normal(0, 0.02)
        event = detector.update(step, train, val)
        if event:
            grok_detected = True
            print(f"  [GROK] Detected at step {event.step}")
            print(f"         Val loss: {event.val_loss_before:.3f} -> {event.val_loss_after:.3f}")
            print(f"         Drop: {event.relative_drop:.1%}")

    assert grok_detected, "Grokking should have been detected!"
    print("  [PASS] Grokking detection working")

    # Test analysis
    analysis = detector.get_analysis()
    assert analysis['status'] == 'grokking_detected'
    print("  [PASS] Analysis output correct")

    print("\n  GrokkingDetector: ALL TESTS PASSED")


def test_metrics_logger():
    """Test metrics logging."""
    print("\n" + "="*60)
    print("Testing MetricsLogger...")
    print("="*60)

    from src.training.metrics_logger import MetricsLogger

    temp_dir = tempfile.mkdtemp()

    try:
        logger = MetricsLogger(
            experiment_name='test_metrics',
            log_dir=temp_dir,
            console_interval=50,
            use_wandb=False,
        )

        # Log some metrics
        for step in range(0, 100, 10):
            logger.log(step, {
                'train_loss': 1.0 - step * 0.005,
                'learning_rate': 0.001,
            })

        # Log eval metrics
        logger.log(100, {'val_loss': 0.6}, eval_metrics=True)
        print("  [PASS] Metrics logged")

        # Get statistics
        stats = logger.get_statistics()
        assert stats['total_steps'] == 100
        assert 'train_loss' in stats['metrics']
        print("  [PASS] Statistics computed")

        # Save
        logger.save_summary()
        print("  [PASS] Summary saved")

        # Check files exist
        assert (logger.log_path / 'metrics.csv').exists()
        assert (logger.log_path / 'metrics.json').exists()
        print("  [PASS] Output files created")

        print("\n  MetricsLogger: ALL TESTS PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("Testing Config...")
    print("="*60)

    from src.utils.config import load_config, ExperimentConfig, print_config

    # Test default config
    config = ExperimentConfig()
    assert config.model.n_layers == 6
    assert config.training.learning_rate == 3e-4
    print("  [PASS] Default config created")

    # Test to_dict
    d = config.to_dict()
    assert 'model' in d
    assert 'training' in d
    print("  [PASS] Config serialization")

    # Test from_dict
    config2 = ExperimentConfig.from_dict(d)
    assert config2.model.n_layers == config.model.n_layers
    print("  [PASS] Config deserialization")

    # Test loading actual config file if exists
    config_path = 'configs/exp_a1_prose_only.yaml'
    if os.path.exists(config_path):
        config3 = load_config(config_path)
        assert config3.name == 'EXP-A1-prose-only'
        print("  [PASS] YAML config loaded")
        print_config(config3)
    else:
        print("  [SKIP] Config file not found (run from project root)")

    print("\n  Config: ALL TESTS PASSED")


def test_device_detection():
    """Test device auto-detection."""
    print("\n" + "="*60)
    print("Testing Device Detection...")
    print("="*60)

    from src.utils.config import get_device, HardwareConfig

    config = HardwareConfig(device='auto')
    device = get_device(config)

    print(f"  Detected device: {device}")
    assert device in ['cuda', 'mps', 'cpu']

    if torch.cuda.is_available():
        print("  [INFO] CUDA available")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  [INFO] MPS (Apple Silicon) available")
    else:
        print("  [INFO] Using CPU")

    print("\n  Device Detection: PASSED")


def run_all_tests():
    """Run all infrastructure tests."""
    print("\n" + "="*60)
    print("SASLM Training Infrastructure Tests")
    print("="*60)

    tests = [
        ("CheckpointManager", test_checkpoint_manager),
        ("GrokkingDetector", test_grokking_detector),
        ("MetricsLogger", test_metrics_logger),
        ("Config", test_config),
        ("Device Detection", test_device_detection),
    ]

    results = []

    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            results.append((name, f"FAIL: {e}"))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)

    for name, result in results:
        status = "[OK]" if result == "PASS" else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED - Infrastructure ready!")
        return 0
    else:
        print("\n  SOME TESTS FAILED - Please fix before training")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
