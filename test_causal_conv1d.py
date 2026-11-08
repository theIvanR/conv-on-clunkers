
#!/usr/bin/env python
"""
Standalone CUDA correctness test for causal-conv1d.

Tests:
    1. CUDA extension import and device availability.
    2. Tensor layout and stride requirements.
    3. Forward convolution against a PyTorch reference.
    4. Final-state output.
    5. Optional SiLU activation.

Usage:
    python test_causal_conv1d.py

Optional:
    python test_causal_conv1d.py --batch 4 --channels 128 --length 1024 --width 4
"""

import argparse
import sys
import traceback

import torch
import torch.nn.functional as F

try:
    import causal_conv1d_cuda
except ImportError as exc:
    print(f"ERROR: Could not import causal_conv1d_cuda: {exc}")
    sys.exit(1)


def report_tensor(name, tensor):
    print(
        f"  {name:8s} "
        f"shape={str(tuple(tensor.shape)):20s} "
        f"stride={str(tensor.stride()):20s} "
        f"dtype={tensor.dtype} "
        f"device={tensor.device}"
    )


def reference_conv1d(x, weight, bias, silu):
    """
    PyTorch reference implementation.

    x:      [B, C, L]
    weight: [C, K]
    bias:   [C]

    Uses left padding to implement causal convolution.
    """
    width = weight.shape[-1]

    y = F.conv1d(
        x,
        weight.unsqueeze(1),
        bias,
        padding=width - 1,
        groups=x.shape[1],
    )

    y = y[..., :x.shape[-1]]

    if silu:
        y = F.silu(y)

    return y


def run_test(batch, channels, length, width, dtype, device):
    print("\n" + "=" * 72)
    print(f"TEST: B={batch}, C={channels}, L={length}, K={width}")
    print(f"      dtype={dtype}, device={device}")
    print("=" * 72)

    # Generate channel-last backing storage, then transpose to [B,C,L].
    x = torch.randn(
        batch, length, channels,
        device=device,
        dtype=dtype,
    ).transpose(1, 2)

    weight = torch.randn(
        channels, width,
        device=device,
        dtype=dtype,
    )

    bias = torch.randn(
        channels,
        device=device,
        dtype=dtype,
    )

    # Output must preserve the input's channel-last layout.
    out = torch.empty_like(x)

    # Final state: [B,C,K-1], channel-last layout.
    final_states = torch.empty(
        batch, width - 1, channels,
        device=device,
        dtype=dtype,
    ).transpose(1, 2)

    print("\nTensor layouts:")
    report_tensor("input", x)
    report_tensor("weight", weight)
    report_tensor("bias", bias)
    report_tensor("output", out)
    report_tensor("final", final_states)

    assert x.stride(1) == 1, "Input channel dimension must have stride 1"
    assert out.stride(1) == 1, "Output channel dimension must have stride 1"
    assert final_states.stride(1) == 1, "Final-state channel dimension must have stride 1"

    # Expected final state: last K-1 input samples.
    expected_final = x[..., -(width - 1):]

    for silu in (False, True):
        print(f"\nRunning CUDA kernel (SiLU={silu})...")

        out.zero_()
        final_states.zero_()

        # Execute the extension.
        causal_conv1d_cuda.causal_conv1d_fwd(
            x,
            weight,
            bias,
            None,           # seq_idx
            None,           # initial_states
            out,            # output
            final_states,   # final_states_out
            silu,           # silu_activation
        )

        # Synchronize to expose asynchronous CUDA errors here.
        torch.cuda.synchronize(device)

        # Compute independent PyTorch reference.
        expected = reference_conv1d(x, weight, bias, silu)

        # Compare output.
        torch.testing.assert_close(
            out,
            expected,
            rtol=2e-2,
            atol=2e-2,
            msg=f"Convolution mismatch (SiLU={silu})",
        )

        # Compare final states.
        torch.testing.assert_close(
            final_states,
            expected_final,
            rtol=0,
            atol=0,
            msg="Final-state mismatch",
        )

        max_error = (out.float() - expected.float()).abs().max().item()

        print(f"  Output max absolute error: {max_error:.6g}")
        print("  Output numerical comparison: PASS")
        print("  Final-state comparison:      PASS")

    print("\nRESULT: PASS")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test causal_conv1d CUDA extension."
    )

    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--width", type=int, default=3)

    args = parser.parse_args()

    if args.width < 2:
        parser.error("--width must be >= 2")

    if args.length < args.width:
        parser.error("--length must be >= --width")

    print("=" * 72)
    print("CAUSAL-CONV1D CUDA TEST")
    print("=" * 72)

    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA runtime:  {torch.version.cuda}")
    print(f"CUDA available:{torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA is unavailable.")
        sys.exit(1)

    device = torch.device("cuda:0")

    print(f"GPU:           {torch.cuda.get_device_name(device)}")
    print(f"Capability:    {torch.cuda.get_device_capability(device)}")

    try:
        run_test(
            batch=args.batch,
            channels=args.channels,
            length=args.length,
            width=args.width,
            dtype=torch.float16,
            device=device,
        )

    except Exception:
        print("\n" + "!" * 72)
        print("TEST FAILED")
        print("!" * 72)
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
