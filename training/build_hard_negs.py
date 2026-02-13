"""
Build hard negatives from SOP dataset (V5).
Mines near-miss candidates for contrastive training.

Usage:
    python training/build_hard_negs.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.bge_m3_embed import BGEM3Embedder
from core.hard_negatives import mine_hard_negatives, save_triplets


def main():
    print("=" * 60)
    print("  Build Hard Negatives for Training")
    print("=" * 60)
    
    # Load SOP data
    aug_file = os.path.join(config.DATA_DIR, "sop_data_augmented.json")
    source_file = aug_file if os.path.exists(aug_file) else config.SOP_DATA_FILE
    
    with open(source_file, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    print(f"📂 Loaded {len(entries)} entries")
    
    # Embed all questions
    embedder = BGEM3Embedder()
    embedder.load()
    
    questions = [e["question"] for e in entries]
    print("🔄 Embedding all questions...")
    result = embedder.embed_batch(questions)
    dense_vecs = result["dense"]
    
    # Mine hard negatives
    print("🔄 Mining hard negatives...")
    triplets = mine_hard_negatives(
        entries=entries,
        dense_vectors=dense_vecs,
        top_k=10,
        max_negatives=3,
    )
    
    # Save
    save_triplets(triplets)
    
    print(f"\n✅ Done! {len(triplets)} hard negative triplets ready for training.")
    print("Next: python training/finetune_bge_m3.py")


if __name__ == "__main__":
    main()
