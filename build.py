"""
Build script — parses data and builds the FAISS index in one step.

Usage:
    python build.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.parse_data import main as parse_data
from core.pipeline import SOPPipeline


def main():
    print("=" * 60)
    print("  SOP Chatbot — Full Build")
    print("=" * 60)

    # Step 1: Parse data
    print("\n📋 STEP 1: Parse & augment SOP data\n")
    parse_data()

    # Step 2: Build index
    print("\n\n📋 STEP 2: Build FAISS index\n")
    pipeline = SOPPipeline()
    pipeline.build()

    print("\n" + "=" * 60)
    print("  ✅ BUILD COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run evaluation:  python evaluate.py")
    print("  2. Launch web UI:   python app.py")


if __name__ == "__main__":
    main()
