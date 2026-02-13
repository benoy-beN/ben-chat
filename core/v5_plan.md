

V4 → V5 MIGRATION (ACCURACY-FIRST) 
WHAT YOU REMOVE / DOWNGRADE

[-] Dual embedding models with fixed weights (0.7 / 0.3)
[-] Fixed similarity threshold (e.g., 0.75 hardcoded)
[-] Retrieval = FAISS Top-1 only
[-] Reranker used without calibrated decision
[-] No lexical retrieval (BM25 missing)
[-] Raw cosine score treated as confidence

WHAT YOU ADD / REPLACE

[+] Single strong retriever: BGE-M3 (dense + sparse)
[+] BM25 lexical retriever (exact token matching)
[+] Learned fusion (query-adaptive weights)
[+] Top-K retrieval (K=20 or 50)
[+] Cross-encoder reranker (Top-K -> Top-1)
[+] Adaptive threshold (learned on validation)
[+] Confidence calibration (Platt / temperature)
[+] Hard negatives in dataset
[+] Evaluation exports (ROC, confusion matrix)
[+] Semantic cache (optional, speed + stability)


 

TARGET V5 ARCHITECTURE (ACCURACY-FIRST)

[ USER QUERY ]
      |
      v
+----------------------------+
| Query Normalize            |
| - lowercase                |
| - punctuation              |
| - SOP term map             |
+----------------------------+
      |
      v
+----------------------------+
| BGE-M3 Embeddings          |
| - Dense vectors            |
| - Sparse vectors           |
+----------------------------+
      |
      +------------------------+
      |                        |
      v                        v
[ Vector Search (FAISS) ]  [ BM25 Lexical Index ]
      |                        |
      +-----------+------------+
                  |
                  v
         +------------------------+
         | Learned Fusion         |
         | (query-adaptive)       |
         +------------------------+
                  |
                  v
         +------------------------+
         | Top-K Candidates       |
         | (K = 20 / 50)          |
         +------------------------+
                  |
                  v
         +------------------------+
         | Cross-Encoder Reranker |
         | (pick best)            |
         +------------------------+
                  |
                  v
         +------------------------+
         | Confidence Calibration |
         | (Platt / Temp scale)   |
         +------------------------+
                  |
        +---------+---------+
        |                   |
   [ ANSWER ]           [ REJECT ]
   + confidence         + clarify


 

REPO STRUCTURE DIFF (v4 → v5)

/ben-chat
  /retrieval
    - dense_embed.py           (REPLACE with bge_m3_embed.py)
    - faiss_index.py          (KEEP)
    + bm25_index.py           (ADD)
    + fusion.py               (ADD: learned fusion model)
    + hard_negatives.py       (ADD)

  /reranker
    - rerank.py               (KEEP)
    + calibrate.py            (ADD: Platt / temp scaling)

  /training
    + finetune_bge_m3.py      (ADD)
    + generate_paraphrases.py (KEEP / IMPROVE)
    + build_hard_negs.py      (ADD)

  /eval
    - evaluate.py             (EXTEND)
    + roc_curve.py            (ADD)
    + confusion_matrix.py     (ADD)

  /runtime
    + semantic_cache.py       (ADD, optional)


 

IMPLEMENTATION CHECKLIST (IN ORDER)

[ ] Replace dual encoders with BGE-M3
[ ] Add BM25 lexical retrieval
[ ] Change Top-1 FAISS to Top-K (20/50)
[ ] Add learned fusion (logistic regression or small MLP)
[ ] Add cross-encoder reranking (Top-K -> Top-1)
[ ] Add adaptive threshold logic
[ ] Calibrate confidence (Platt scaling)
[ ] Add hard negatives to training data
[ ] Export ROC + confusion matrix in eval
[ ] Add semantic cache (optional)


 

EXPECTED ACCURACY JUMP (REALISTIC)

v4 current:               ~92%
+ BM25:                   ~94–95%
+ BGE-M3:                 ~95–96%
+ Adaptive threshold:     ~96–97%
+ Hard-negative tuning:   ~97–99% (small SOP domains benefit a lot)


 

ONE-LINE MIGRATION PRINCIPLE

STOP tuning infra.
FIX retrieval quality.
CALIBRATE decisions.
MEASURE everything.

