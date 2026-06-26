import numpy as np
from app.utils.complexity_eval import evaluate_complexity


def test_evaluate_complexity_empty_inputs():
    assert evaluate_complexity(None) == 0.0
    assert evaluate_complexity(np.array([])) == 0.0
    assert evaluate_complexity(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_evaluate_complexity_flat_images():
    # Solid black
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert evaluate_complexity(black_img) == 0.0

    # Solid white
    white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert evaluate_complexity(white_img) == 0.0

    # Uniform gray
    gray_img = np.ones((100, 100), dtype=np.uint8) * 128
    assert evaluate_complexity(gray_img) == 0.0


def test_evaluate_complexity_gradients():
    # Create a smooth vertical gradient: variance will be non-zero,
    # but Canny edges will be zero. Gradient from 100 to 120.
    grad = np.zeros((100, 100), dtype=np.uint8)
    for y in range(100):
        grad[y, :] = 100 + int(y * 20 / 100)

    score = evaluate_complexity(grad)
    # The score should be less than 0.05 (mostly flat/empty)
    assert score < 0.05


def test_evaluate_complexity_high_frequency():
    # Create an image with high frequency detail: random white noise
    np.random.seed(42)
    noise_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    score = evaluate_complexity(noise_img)
    # White noise should have high complexity, definitely > 0.05
    assert score > 0.05

    # Check color image noise
    noise_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    assert evaluate_complexity(noise_color) > 0.05


def test_evaluate_complexity_non_uint8():
    # Float image with a gentle gradient in [0.4, 0.5]
    grad_float = np.zeros((100, 100), dtype=np.float32)
    for y in range(100):
        grad_float[y, :] = 0.4 + (y / 100.0) * 0.1
    score = evaluate_complexity(grad_float)
    assert score < 0.05

    # Uniform float
    flat_float = np.ones((100, 100), dtype=np.float32) * 0.5
    assert evaluate_complexity(flat_float) == 0.0
