import random
import os

def split_data():
    input_file = "eval.txt"
    calib_file = "calib.txt"
    test_file = "test.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse blocks
    blocks = []
    current_block = []
    for line in content.splitlines():
        if line.strip():
            current_block.append(line)
        else:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
    if current_block:
        blocks.append("\n".join(current_block))

    print(f"Total blocks found: {len(blocks)}")
    
    # Shuffle and split
    random.seed(42) # Deterministic split
    random.shuffle(blocks)
    
    split_idx = int(len(blocks) * 0.7) # 70/30 split for V6.3 (more calibration data)
    calib_data = blocks[:split_idx]
    test_data = blocks[split_idx:]
    
    print(f"Calibration set: {len(calib_data)}")
    print(f"Test set: {len(test_data)}")
    
    with open(calib_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(calib_data))
        
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(test_data))

    print("✅ Data split complete.")

if __name__ == "__main__":
    split_data()
