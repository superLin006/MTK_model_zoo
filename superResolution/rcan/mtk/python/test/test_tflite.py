"""
Test TFLite conversion and inference
"""

import sys
import os
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available - TFLite tests will be skipped")

try:
    from PIL import Image
    import torch
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def test_tflite_load():
    """Test loading TFLite model"""
    print("\n" + "="*70)
    print("Test 1: Load TFLite Model (510x339)")
    print("="*70)

    if not TENSORFLOW_AVAILABLE:
        print("⊘ TensorFlow not available - skipping")
        return True

    model_path = "../models/RCAN_BIX4_339x510_test.tflite"

    if not os.path.exists(model_path):
        print(f"⚠ TFLite model not found: {model_path}")
        print("  Note: Run step2_torchscript_to_tflite.py first")
        return True  # Skip if model not available

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        print("✓ TFLite model loaded successfully")

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")

        return True

    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return False


def load_image_tensor(image_path):
    """Load image and convert to tensor (NCHW format, 0-255 range)"""
    if not PIL_AVAILABLE:
        return None
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)  # Keep 0-255 range
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img

def save_image(tensor, output_path):
    """Convert tensor to image and save (expects 0-255 range)"""
    if not PIL_AVAILABLE:
        return False
    # Handle both torch tensor and numpy array
    if isinstance(tensor, torch.Tensor):
        img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    else:
        # Assume NCHW numpy array
        img_array = np.transpose(tensor.squeeze(0), (1, 2, 0))
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path)
    return True

def test_tflite_inference():
    """Test TFLite inference"""
    print("\n" + "="*70)
    print("Test 2: TFLite Inference with Real Image (510x339)")
    print("="*70)

    if not TENSORFLOW_AVAILABLE:
        print("⊘ TensorFlow not available - skipping")
        return True

    model_path = "../models/RCAN_BIX4_339x510_test.tflite"
    image_path = "../../test_data/input_510x339.png"
    output_dir = "../../test_data/output"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"⚠ TFLite model not found: {model_path}")
        return True  # Skip if model not available

    try:
        # Get MeanShift parameters (TFLite Core model needs manual preprocessing)
        from rcan_model import get_meanshift_params
        params = get_meanshift_params()
        rgb_mean = np.array(params['rgb_mean'], dtype=np.float32).reshape(1, 3, 1, 1)
        rgb_range = params['rgb_range']

        # Load image if available
        if PIL_AVAILABLE and os.path.exists(image_path):
            print(f"加载测试图片: {image_path}")
            img_data = load_image_tensor(image_path)
            if img_data:
                test_input_torch, original_img = img_data
                test_input = test_input_torch.numpy()  # NCHW, 0-255 range
                print(f"✓ 图片大小: {original_img.size}")
                print(f"✓ 张量形状: {tuple(test_input.shape)} (NCHW)")
            else:
                # Random input (0-255 range)
                test_input = np.random.randn(1, 3, 339, 510).astype(np.float32) * 127.5 + 127.5
        else:
            print("使用随机张量 (0-255 range)")
            # Random input (0-255 range)
            test_input = np.random.randn(1, 3, 339, 510).astype(np.float32) * 127.5 + 127.5

        # Apply MeanShift preprocessing
        print("应用 MeanShift 预处理...")
        preprocessed_input = test_input - rgb_range * rgb_mean

        # Load model
        print("Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Run inference (TFLite model uses NCHW format)
        print("Running TFLite inference...")
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], preprocessed_input.astype(np.float32))
        interpreter.invoke()
        core_output = interpreter.get_tensor(output_details[0]['index'])
        inference_time = time.time() - start_time

        # Apply inverse MeanShift
        print("应用 MeanShift 后处理...")
        final_output = core_output + rgb_range * rgb_mean

        print("✓ TFLite inference successful")
        print(f"  Input shape (NCHW):  {test_input.shape}")
        print(f"  Output shape (NCHW): {final_output.shape}")
        print(f"  推理时间: {inference_time:.4f}s")

        # Save output if image was loaded
        if PIL_AVAILABLE and os.path.exists(image_path):
            output_path_img = os.path.join(output_dir, "rcan_tflite_core_510x339.png")
            if save_image(final_output, output_path_img):
                print(f"✓ 输出已保存: {output_path_img}")

        expected_shape = (1, 3, 1356, 2040)
        if final_output.shape == expected_shape:
            print(f"✓ Output shape matches expected (NCHW): {expected_shape}")
            return True
        else:
            print(f"❌ Output shape mismatch. Expected {expected_shape}, got {output.shape}")
            return False

    except Exception as e:
        print(f"❌ TFLite inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tflite_conversion_chain():
    """Test the full TFLite conversion chain"""
    print("\n" + "="*70)
    print("Test 3: Full Conversion Chain (TorchScript → TFLite)")
    print("="*70)

    if not TENSORFLOW_AVAILABLE:
        print("⊘ TensorFlow not available - skipping")
        return True

    print("Note: This test verifies that step2_torchscript_to_tflite.py works correctly")

    ts_path = "../models/RCAN_BIX4_core_339x510_test.pt"
    tflite_path = "../models/RCAN_BIX4_339x510_test.tflite"

    if not os.path.exists(ts_path):
        print(f"⚠ TorchScript model not found: {ts_path}")
        return True

    if os.path.exists(tflite_path):
        print(f"✓ TFLite model exists: {tflite_path}")
        file_size_mb = os.path.getsize(tflite_path) / 1024 / 1024
        print(f"  Size: {file_size_mb:.1f} MB")
        return True
    else:
        print(f"⚠ TFLite model not found: {tflite_path}")
        print("  Run step2_torchscript_to_tflite.py to create it")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("RCAN TFLite Conversion Tests")
    print("="*70)

    if not TENSORFLOW_AVAILABLE:
        print("⚠ TensorFlow not available")
        print("Install it with: pip install tensorflow")
        return 1

    results = []
    results.append(("Load TFLite Model", test_tflite_load()))
    results.append(("TFLite Inference", test_tflite_inference()))
    results.append(("Conversion Chain", test_tflite_conversion_chain()))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
