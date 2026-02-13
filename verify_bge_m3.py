import sys
import os
import torch
import numpy as np

# Ensure we can import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.bge_m3_embed import BGEM3Embedder
from core.bge_m3_embed import BGEM3Embedder
import config

def verify():
    print("🔍 Initializing BGEM3Embedder...")
    embedder = BGEM3Embedder()
    
    # Load model (forces load)
    embedder.load()
    
    model = embedder.model
    print(f"\n✅ Model Object: {type(model)}")
    print(f"✅ Model config name/path: {embedder.model_name}")
    
    # Check dimensions
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
        print(f"📏 Reported Dimension: {dim}")
        if dim == 1024:
            print("   (Matches BGE-M3 spec)")
        else:
            print(f"   ⚠️ MISMATCH! BGE-M3 should be 1024. Got {dim}")
    
    # Check underlying transformer config
    try:
        # SentenceTransformers wraps Transformer in .0 (usually) or modules
        transformer = model._first_module()
        config = transformer.auto_model.config
        print(f"⚙️  Transformer Config: {config.name_or_path}")
        print(f"    - Hidden Size: {config.hidden_size}")
        print(f"    - Vocab Size: {config.vocab_size}")
        print(f"    - Layers: {config.num_hidden_layers}")
    except Exception as e:
        print(f"    (Could not inspect inner transformer: {e})")

    # Real inference test
    test_text = "Verify this embedding."
    print(f"\n🧪 Running Inference on: '{test_text}'")
    vec = embedder.embed(test_text)["dense"]
    
    print(f"📊 Output Vector Shape: {vec.shape}")
    print(f"    - First 5 values: {vec[:5]}")
    
    if vec.shape[0] == 1024:
        print("\n✅ VERIFICATION SUCCESS: Real BGE-M3 (1024d) is active.")
    else:
        print("\n❌ VERIFICATION FAILED: Dimension mismatch.")

if __name__ == "__main__":
    verify()
