#!/usr/bin/env python3
import os
import struct
import math
import array

def inspect_multibeam_file(filepath, max_beams=4):
    print("==================================================")
    print(f"Inspecting file: {filepath}")
    file_size = os.path.getsize(filepath)
    print(f"File size on disk: {file_size:,} bytes")
    
    with open(filepath, "rb") as f:
        meta_size_data = f.read(4)
        if len(meta_size_data) < 4:
            print("Error: file too small")
            return
        meta_size = struct.unpack("I", meta_size_data)[0]
        print(f"Metadata header size: {meta_size} bytes")
        if meta_size > 0:
            meta_bytes = f.read(meta_size)
            print(f"Metadata read: {len(meta_bytes)} bytes")
        
        payload_bytes = f.read()
        print(f"Payload size: {len(payload_bytes):,} bytes")
        
        # Read float32 array
        floats = array.array('f')
        floats.frombytes(payload_bytes)
        n_samples = len(floats)
        expected_samples = 15360 * 336 * max_beams
        
        print(f"Total float32 elements: {n_samples:,} (Expected: {expected_samples:,} for {max_beams} beams)")
        
        if n_samples != expected_samples:
            print(f"[WARNING] Sample count mismatch! Expected {expected_samples}, got {n_samples}")
            return
        
        # Check active beam 0 (first beam)
        min_val = float('inf')
        max_val = float('-inf')
        total_sum = 0.0
        nan_count = 0
        inf_count = 0
        negative_count = 0
        
        # Stride through beam 0 elements
        beam0_count = 15360 * 336
        for idx in range(0, n_samples, max_beams):
            val = floats[idx]
            if math.isnan(val):
                nan_count += 1
            elif math.isinf(val):
                inf_count += 1
            else:
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
                if val < 0.0:
                    negative_count += 1
                total_sum += val
                
        mean_val = total_sum / beam0_count
        
        print(f"\n--- Beam 0 Numerical Statistics ---")
        print(f"  Min Intensity   : {min_val:.4f}")
        print(f"  Max Intensity   : {max_val:.4f}")
        print(f"  Mean Intensity  : {mean_val:.4f}")
        print(f"  Contains NaN?   : {'YES [FAIL]' if nan_count > 0 else 'NO [PASS]'}")
        print(f"  Contains Inf?   : {'YES [FAIL]' if inf_count > 0 else 'NO [PASS]'}")
        print(f"  Non-negative?   : {'NO [FAIL]' if negative_count > 0 else 'YES [PASS]'}")
        
        print("\nSample first 8 Intensity values (time=0, freq=0..1, beams 0..3):")
        print([round(floats[i], 4) for i in range(8)])
        print("==================================================\n")

if __name__ == "__main__":
    files = [
        "/tmp/test_charts_tracker/multibeam_intensity_0000000.bin",
        "/tmp/test_charts_tracker/multibeam_intensity_0000001.bin"
    ]
    for f in files:
        if os.path.exists(f):
            inspect_multibeam_file(f, max_beams=4)
        else:
            print(f"File not found: {f}")
