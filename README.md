# Machine Learning Interview Preparation Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Author: Arun Brahma](https://img.shields.io/badge/Author-Arun%20Brahma-purple)](https://github.com/iamarunbrahma)

Preparing for a Machine Learning (ML) interview is easier with a focused plan and high-signal resources. This guide prioritizes free, reputable material and gives you a practical roadmap for coding, ML breadth, system design, and modern domains (DL, NLP, CV, RecSys, LLMs/RAG/Agents).

## What to expect (Big Tech vs Startups)

- Big Tech: more rounds, consistent rubrics; strong emphasis on ML system design, data/metrics rigor, and coding under time pressure.
- Startups: fewer rounds, more applied/product focus; expect deeper dives into end-to-end ownership, scrappy experimentation, and shipping impact quickly.
- For both: prepare for coding, ML concepts, system design, and behavioral. Bring 2–3 projects you can whiteboard end-to-end.

## Coding (DSA)

LeetCode is widely used for screening. Practice mixed difficulty and focus on patterns.

- Target distribution: ~55% Medium, ~35% Easy, ~10–15% Hard. Track via [LeetCode Study Plan](https://leetcode.com/studyplan/top-interview-150/).
- Pattern-based free prep:
  - [seanprashad/leetcode-patterns](https://github.com/seanprashad/leetcode-patterns)
  - [yangshun/tech-interview-handbook](https://github.com/yangshun/tech-interview-handbook)
- Rehearse: solve core Mediums 2–3 times to lock in speed and pattern recognition.

## ML System Design

> [!IMPORTANT]
> Start with business goals and success metrics. Then walk data → features → training → evaluation → serving → monitoring. Make trade-offs and failure modes explicit.

Understand components and trade-offs from data to deployment; ground your answers in real case studies; communicate clearly and concisely.

- Primer and frameworks (free):
  - [chiphuyen/machine-learning-systems-design](https://github.com/chiphuyen/machine-learning-systems-design)
  - [khangich/machine-learning-interview (design.md)](https://github.com/khangich/machine-learning-interview/blob/master/design.md)
  - [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews)
- Case studies to ground designs:
  - [Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies](https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies)
  - [mallahyari/ml-practical-usecases](https://github.com/mallahyari/ml-practical-usecases)
  - [themanojdesai/genai-llm-ml-case-studies](https://github.com/themanojdesai/genai-llm-ml-case-studies)
- Production lessons:
  - [eugeneyan/applied-ml](https://github.com/eugeneyan/applied-ml)

Prefer simple baselines first; iterate; discuss trade-offs, failure modes, and how you’ll measure and improve.

## ML Coding (from scratch and with libraries)

- Implement classics from scratch (e.g., linear/logistic regression, k-means, k-NN, decision trees, basic NN) to solidify fundamentals. Try: [eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- Be fluent with scikit-learn, PyTorch, and/or TensorFlow for fast prototyping.
- Always discuss trade-offs (bias/variance, data/feature constraints, compute, latency).

## ML Concepts (theory and practice)

- Core topics: bias–variance trade-off, regularization, cross-validation, model selection, calibration, class imbalance, drift.
- Concise refreshers:
  - [ml-cheatsheet.readthedocs.io](https://ml-cheatsheet.readthedocs.io/en/latest/)

## Deep Learning

- Question banks and notes:
  - [scutan90/DeepLearning-500-questions](https://github.com/scutan90/DeepLearning-500-questions)
  - [Devinterview-io/deep-learning-interview-questions](https://github.com/Devinterview-io/deep-learning-interview-questions)

## Natural Language Processing (NLP)

- Question banks and checklists:
  - [Devinterview-io/nlp-interview-questions](https://github.com/Devinterview-io/nlp-interview-questions)
  - [spidey1202/NLP-interview-questions](https://github.com/spidey1202/NLP-interview-questions)

## Computer Vision (CV)

- Question banks:
  - [Devinterview-io/computer-vision-interview-questions](https://github.com/Devinterview-io/computer-vision-interview-questions)
  - [Praveen76/Computer-Vision-Interview-Preparation](https://github.com/Praveen76/Computer-Vision-Interview-Preparation)

## Recommender Systems

- Focus areas: collaborative vs content-based vs hybrid; matrix factorization (SVD), implicit feedback, cold start, evaluation (MAP/NDCG/AUC), online metrics.
- Free Q/A:
  - [Devinterview-io/recommendation-systems-interview-questions](https://github.com/Devinterview-io/recommendation-systems-interview-questions)

## Generative AI and LLMs

- Interview-focused:
  - [Devinterview-io/llms-interview-questions](https://github.com/Devinterview-io/llms-interview-questions)
  - [llmgenai/LLMInterviewQuestions](https://github.com/llmgenai/LLMInterviewQuestions)
- Ecosystem overview and examples (agents, RAG, apps):
  - [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)

## RAG (Retrieval-Augmented Generation)

- Curated lists:
  - [Danielskry/Awesome-RAG](https://github.com/Danielskry/Awesome-RAG)
  - [coree/awesome-rag](https://github.com/coree/awesome-rag)

## Agentic AI (tools, planning, multi-step autonomy)

- Hands-on and catalogs:
  - [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)

## Big-picture preparation tips

Practice whiteboarding with structure and timeboxes; favor clarity over exhaustive detail.
Keep a personal “wins and failures” log to support behavioral answers with measurable impact.
Build and demo 1–2 small end-to-end projects (e.g., ranking, anomaly detection, simple RAG chatbot) and be ready to discuss design and trade-offs.

## Curated free, reputable ML interview guides

- Alireza Dirafzoon’s ML Interviews: https://github.com/alirezadir/Machine-Learning-Interviews
- Khang Nguyen’s ML Interview (incl. design): https://github.com/khangich/machine-learning-interview
- Andrew Khalel MLQuestions (ML + CV Qs): https://github.com/andrewekhalel/MLQuestions
- Khanh Nam Le “Cracking the DS Interview”: https://github.com/khanhnamle1994/cracking-the-data-science-interview
- Rishabh Bhatia DS Interview Resources: https://github.com/rbhatia46/Data-Science-Interview-Resources
- Deep Learning Interviews (free arXiv PDF): https://github.com/BoltzmannEntropy/interviews.ai

## End-to-end personal projects (free Kaggle)

- H&M Personalized Fashion Recommendations (RecSys, retrieval + ranking): https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
- IEEE-CIS Fraud Detection (large-scale, heavy class imbalance): https://www.kaggle.com/competitions/ieee-fraud-detection
- Google Analytics Customer Revenue Prediction (forecasting/regression, leakage-aware CV): https://www.kaggle.com/c/ga-customer-revenue-prediction
- M5 Forecasting – Accuracy (hierarchical time series forecasting): https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Mercari Price Suggestion (tabular + text, feature engineering at scale): https://www.kaggle.com/competitions/mercari-price-suggestion-challenge

Project deliverables to aim for

- Clear problem definition and success metrics
- Clean feature pipeline (document point-in-time correctness)
- Offline evaluation (appropriate metrics) and a simple online plan
- Lightweight serving stub (batch or API) and monitoring checklist

## 30-day preparation plan

- Format: 6 days/week + 1 lighter buffer/review day. Each day ~2–4 hours. Mix DSA, ML concepts/coding, system design, and one domain. Keep notes and practice aloud.

Week 1 (Foundations + Patterns)

- Day 1: DSA arrays/two-pointers; ML concepts: bias–variance, over/underfitting; Domain: DL basics (activation, loss); Read: chiphuyen booklet intro
- Day 2: DSA sliding window; ML coding: linear/logistic regression from scratch; Domain: NLP basics (tokenization, TF-IDF)
- Day 3: DSA hashing; ML concepts: regularization, calibration, class imbalance; Domain: CV basics (edges, HOG)
- Day 4: DSA stacks/queues; System design: metrics, goals, constraints; Case study skim (Engineer1999)
- Day 5: DSA binary search; ML coding: k-NN, k-means; Domain: RecSys paradigms (CF/content/hybrid)
- Day 6: Timed practice session (30m coding + 30m ML breadth); review and fix weak spots
- Day 7: Light buffer: 1–2 LeetCode mediums

Week 2 (Graphs + System Design)

- Day 8: DSA BFS/DFS; ML concepts: CV/holdout/K-fold; Domain: LLM basics (tokenization, embeddings)
- Day 9: DSA trees/BST; System design: data pipeline, features, training; Case study: ranking/search
- Day 10: DSA heaps/priority queues; ML coding: decision trees; Domain: CV CNNs
- Day 11: DSA intervals/greedy; System design: serving, latency/throughput/SLO; Introduce monitoring
- Day 12: DSA backtracking; Domain: NLP sequence models, attention; LLM evaluation basics
- Day 13: System design scenario drill (45m); review
- Day 14: Light buffer: 1 domain deep-dive reading + 1 LC medium

Week 3 (Optimization + Domain Rotations)

- Day 15: DSA graph shortest paths; ML concepts: metrics (PR/ROC/AUC/NDCG/MAP); Domain: RecSys MF/SVD
- Day 16: DSA union-find; System design: offline vs online eval, A/B, bandits
- Day 17: ML coding: PyTorch quickstart; Domain: LLM RAG basics (chunking, vector DB)
- Day 18: DSA practice set (4 mediums); Domain: Agentic patterns; Monitoring hallucinations
- Day 19: Projects polish: small E2E demo (ranking or RAG chatbot); Prepare talking points
- Day 20: Integrated timed session (coding + ML breadth); review
- Day 21: Light buffer: review errors log + 1 LC hard attempt

Week 4 (Polish + Drills)

- Day 22: System design dry runs (recap structure); Case studies scan (genAI/LLM)
- Day 23: DSA mixed timed set (3 mediums); ML coding: pipeline hygiene
- Day 24: Domain pick 1 deep dive: DL or NLP; summarize 5 Q/A
- Day 25: Domain pick 2 deep dive: CV or RecSys; summarize 5 Q/A
- Day 26: Domain pick 3 deep dive: LLM/RAG/Agentic; summarize 5 Q/A
- Day 27: Full-cycle dry run (coding + system design); review
- Day 28: Behavioral prep (STAR stories x6); metrics/impact framing
- Day 29: Final gaps: error log pass + 2 mediums; sleep routine
- Day 30: Light review only; logistics; hydrate + rest

Deliverables each day

- 2–4 LC problems (pattern-aligned); 5–10 lines of notes; 1 case study bullet takeaway on SD days.

## One-page printable checklist

- [ ] Coding (DSA)
  - [ ] Arrays / two-pointers
  - [ ] Sliding window
  - [ ] Hashing
  - [ ] Stacks / queues
  - [ ] Binary search
  - [ ] Heaps
  - [ ] Trees / BST
  - [ ] Graphs (BFS / DFS)
  - [ ] Intervals / greedy
  - [ ] Backtracking
  - [ ] DP basics
  - [ ] Timed practice (2–4 problems/day; redo misses)

- [ ] ML Concepts
  - [ ] Bias–variance trade-off
  - [ ] Regularization (L1/L2, dropout, early stopping)
  - [ ] Cross-validation, data leakage avoidance
  - [ ] Metrics: PR/ROC/AUC, F1, log-loss, calibration
  - [ ] Class imbalance handling
  - [ ] Drift (data/concept) detection

- [ ] ML Coding
  - [ ] From-scratch: linear/logistic regression
  - [ ] From-scratch: k-NN, k-means
  - [ ] From-scratch: decision trees
  - [ ] scikit-learn pipelines
  - [ ] PyTorch quickstart

- [ ] ML System Design
  - [ ] Goals, constraints, metrics (online/offline)
  - [ ] Data → features → training → eval → serving → monitoring
  - [ ] Latency/throughput/cost trade-offs
  - [ ] A/B tests, bandits, guardrails

- [ ] Domains
  - [ ] DL: activations, losses, CNN/RNN/Transformers
  - [ ] NLP: tokenization, embeddings, sequence models
  - [ ] CV: edges/HOG, CNN basics
  - [ ] RecSys: CF/content/hybrid, SVD, NDCG/MAP
  - [ ] LLM/GenAI: embeddings, evaluation
  - [ ] RAG: chunking, vector DB, retrieval, re-ranking
  - [ ] Agentic: tools/memory/planning; safety

- [ ] Case Studies (pick 5)
  - [ ] Use case, metrics, architecture, lessons

- [ ] Projects (1–2 E2E demos)
  - [ ] Ranking or anomaly detection
  - [ ] Simple RAG chatbot
  - [ ] Talk track: problem → data → model → eval → prod → impact

- [ ] Behavioral
  - [ ] STAR stories ×6 with metrics and failure lessons



- [ ] Logistics
  - [ ] Environment setup, whiteboard kit
  - [ ] Calendar blocks, rest plan

## Contribute & share

⭐ Star this repo, share it with a friend, and open a PR if you have a great free resource to add.

## Attribution

All external links above are freely accessible at the time of writing. If a link becomes paid or unavailable, please open an issue or PR to update it.
