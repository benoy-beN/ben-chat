"""
Fine-tune BGE-M3 on SOP data with hard negatives (V5).
Uses FlagEmbedding's training API for contrastive learning.

Usage:
    python training/finetune_bge_m3.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_training_data():
    """Load hard negatives and SOP data for fine-tuning."""
    hard_neg_path = os.path.join(config.DATA_DIR, "hard_negatives.json")
    
    if not os.path.exists(hard_neg_path):
        print("⚠️  No hard negatives found. Run build_hard_negs.py first.")
        return []
    
    with open(hard_neg_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)
    
    # Convert to training format: {"query": ..., "pos": [...], "neg": [...]}
    training_data = []
    for t in triplets:
        training_data.append({
            "query": t["anchor"],
            "pos": [t["positive"]],
            "neg": [t["negative"]],
        })
    
    return training_data


def finetune():
    """Fine-tune BGE-M3 on SOP training data."""
    print("=" * 60)
    print("  Fine-tune BGE-M3 on SOP Data")
    print("=" * 60)
    
    training_data = load_training_data()
    if not training_data:
        print("❌ No training data. Exiting.")
        return
    
    print(f"📂 Loaded {len(training_data)} training examples")
    
    # Save training data in the format FlagEmbedding expects
    train_file = os.path.join(config.DATA_DIR, "finetune_train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"💾 Training file saved → {train_file}")
    
    try:
        from FlagEmbedding import FlagModel
        
        # Note: Full fine-tuning requires FlagEmbedding's training scripts
        # This is a template — for production, use:
        #   python -m FlagEmbedding.baai_general_embedding.finetune.run \
        #     --model_name_or_path BAAI/bge-m3 \
        #     --train_data {train_file} \
        #     --output_dir {output_dir} \
        #     --num_train_epochs 3 \
        #     --per_device_train_batch_size 4 \
        #     --learning_rate 1e-5
        
        output_dir = os.path.join(config.DATA_DIR, "finetuned_bge_m3")
        
        print("\n📋 To fine-tune BGE-M3, run:")
        print(f"   python -m FlagEmbedding.baai_general_embedding.finetune.run \\")
        print(f"     --model_name_or_path BAAI/bge-m3 \\")
        print(f"     --train_data {train_file} \\")
        print(f"     --output_dir {output_dir} \\")
        print(f"     --num_train_epochs 3 \\")
        print(f"     --per_device_train_batch_size 4 \\")
        print(f"     --learning_rate 1e-5")
        print(f"\n   Then update config.EMBEDDING_MODEL to point to:")
        print(f"   {output_dir}")

    except ImportError:
        print("⚠️  FlagEmbedding not installed. Install with:")
        print("   pip install FlagEmbedding")


if __name__ == "__main__":
    finetune()
