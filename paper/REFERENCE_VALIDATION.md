# Reference validation ledger

All 45 bibliography entries in `references.bib` were checked on 2026-11-17
against the linked primary record: the publisher or proceedings page, the
official OpenReview record, the authors' primary publication page, or arXiv.
For each item, the title, author list, publication year, and venue or report
identifier were compared with that record. A live URL audit then returned a
successful response for 43 records. The PNAS and Project Euclid pages returned
HTTP 403 to the command-line client, so those two records were opened and
checked through the web interface instead. Both resolved to the expected
publisher pages.

Every entry below is cited in the manuscript. The repository test
`tests/test_paper_references.py` enforces a minimum of 40 entries, exact
bibliography-to-citation coverage, a URL for every entry, and a ledger record
for every key.

| Citation key | Work | Authoritative record | Status |
|---|---|---|---|
| `vaswani2017attention` | Attention Is All You Need | https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html | Validated 2026-11-17 |
| `radford2019language` | Language Models are Unsupervised Multitask Learners | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | Validated 2026-11-17 |
| `elhage2021mathematical` | A Mathematical Framework for Transformer Circuits | https://transformer-circuits.pub/2021/framework/index.html | Validated 2026-11-17 |
| `olsson2022induction` | In-context Learning and Induction Heads | https://arxiv.org/abs/2209.11895 | Validated 2026-11-17 |
| `wang2022ioi` | Interpretability in the Wild | https://openreview.net/forum?id=NpsVSN6o4ul | Validated 2026-11-17 |
| `meng2022locating` | Locating and Editing Factual Associations in GPT | https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html | Validated 2026-11-17 |
| `goldowskydill2023path` | Localizing Model Behavior with Path Patching | https://arxiv.org/abs/2304.05969 | Validated 2026-11-17 |
| `zhang2024activation` | Towards Best Practices of Activation Patching | https://openreview.net/forum?id=Hf17y6u9BC | Validated 2026-11-17 |
| `heimersheim2024activation` | How to Use and Interpret Activation Patching | https://arxiv.org/abs/2404.15255 | Validated 2026-11-17 |
| `hase2023localization` | Does Localization Inform Editing? | https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html | Validated 2026-11-17 |
| `makelov2024subspace` | Is This the Subspace You Are Looking for? | https://openreview.net/forum?id=Ebt7JgMHv1 | Validated 2026-11-17 |
| `conmy2023acdc` | Towards Automated Circuit Discovery | https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html | Validated 2026-11-17 |
| `kramar2024atp` | AtP*: Efficient and Scalable Localization | https://arxiv.org/abs/2403.00745 | Validated 2026-11-17 |
| `syed2024attribution` | Attribution Patching Outperforms Automated Circuit Discovery | https://aclanthology.org/2024.blackboxnlp-1.25/ | Validated 2026-11-17 |
| `lieberum2023circuitsscale` | Does Circuit Analysis Interpretability Scale? | https://arxiv.org/abs/2307.09458 | Validated 2026-11-17 |
| `geiger2021causalabstractions` | Causal Abstractions of Neural Networks | https://proceedings.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html | Validated 2026-11-17 |
| `geiger2023causal` | Causal Abstraction: A Theoretical Foundation | https://www.jmlr.org/papers/v26/23-0058.html | Validated 2026-11-17 |
| `geiger2024das` | Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations | https://proceedings.mlr.press/v236/geiger24a.html | Validated 2026-11-17 |
| `vig2020causal` | Investigating Gender Bias Using Causal Mediation Analysis | https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html | Validated 2026-11-17 |
| `wu2024pyvene` | pyvene | https://aclanthology.org/2024.naacl-demo.16/ | Validated 2026-11-17 |
| `haklay2025position` | Position-aware Automatic Circuit Discovery | https://aclanthology.org/2025.acl-long.141/ | Validated 2026-11-17 |
| `geva2021transformer` | Transformer Feed-Forward Layers Are Key-Value Memories | https://aclanthology.org/2021.emnlp-main.446/ | Validated 2026-11-17 |
| `geva2022transformer` | Transformer Feed-Forward Layers Build Predictions | https://aclanthology.org/2022.emnlp-main.3/ | Validated 2026-11-17 |
| `geva2023dissecting` | Dissecting Recall of Factual Associations | https://aclanthology.org/2023.emnlp-main.751/ | Validated 2026-11-17 |
| `elhage2022superposition` | Toy Models of Superposition | https://www.transformer-circuits.pub/2022/toy_model/index.html | Validated 2026-11-17 |
| `huben2024sparse` | Sparse Autoencoders Find Highly Interpretable Features | https://proceedings.iclr.cc/paper_files/paper/2024/hash/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Abstract-Conference.html | Validated 2026-11-17 |
| `templeton2024scaling` | Scaling Monosemanticity | https://www.transformer-circuits.pub/2024/scaling-monosemanticity/index.html | Validated 2026-11-17 |
| `marks2025sparsecircuits` | Sparse Feature Circuits | https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ba4d47a83e498c2b1a0868cba20f6de-Abstract-Conference.html | Validated 2026-11-17 |
| `park2024linear` | The Linear Representation Hypothesis and Geometry of LLMs | https://proceedings.mlr.press/v235/park24c.html | Validated 2026-11-17 |
| `michel2019sixteen` | Are Sixteen Heads Really Better than One? | https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html | Validated 2026-11-17 |
| `pearl2001direct` | Direct and Indirect Effects | https://mlanthology.org/uai/2001/pearl2001uai-direct/ | Validated 2026-11-17 |
| `imai2010general` | A General Approach to Causal Mediation Analysis | https://pubmed.ncbi.nlm.nih.gov/20954780/ | Validated 2026-11-17 |
| `lin1991divergence` | Divergence Measures Based on the Shannon Entropy | https://ieeexplore.ieee.org/document/61115 | Validated 2026-11-17 |
| `efron1979bootstrap` | Bootstrap Methods: Another Look at the Jackknife | https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full | Validated in web interface 2026-11-17 |
| `nosek2018preregistration` | The Preregistration Revolution | https://www.pnas.org/doi/10.1073/pnas.1708274114 | Validated in web interface 2026-11-17 |
| `pineau2021improving` | Improving Reproducibility in Machine Learning Research | https://www.jmlr.org/papers/v22/20-303.html | Validated 2026-11-17 |
| `bouthillier2021accounting` | Accounting for Variance in Machine Learning Benchmarks | https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html | Validated 2026-11-17 |
| `mcgrath2023hydra` | The Hydra Effect | https://arxiv.org/abs/2307.15771 | Validated 2026-11-17 |
| `mueller2024missed` | Missed Causes and Ambiguous Effects | https://openreview.net/forum?id=pJs3ZiKBM5 | Validated 2026-11-17 |
| `miller2024faithfulness` | Transformer Circuit Faithfulness Metrics are not Robust | https://openreview.net/forum?id=zSf8PJyQb2 | Validated 2026-11-17 |
| `olivieri2026field` | Transformer Field Theory | https://arxiv.org/abs/2605.25225 | Validated 2026-11-17 |
| `vaidyanathan2026mediators` | The Curse of Multiple Mediators | https://arxiv.org/abs/2606.27510 | Validated 2026-11-17 |
| `gong2026coax` | Conditional Co-Ablation | https://arxiv.org/abs/2607.01940 | Validated 2026-11-17 |
| `vaidyanathan2026hydra` | Slaying the Hydra | https://arxiv.org/abs/2610.04017 | Validated 2026-11-17 |
| `tanyel2026whose` | Whose Effect Is It? | https://arxiv.org/abs/2611.05033 | Validated 2026-11-17 |

## Corrections made during validation

- The Olivieri and P\'erez Rodr\'iguez title was corrected to the title on the
  arXiv record: *Transformer Field Theory: A Response-Theoretic Approach to
  Mechanistic Interpretability*.
- The Zhang and Nanda entry now links to the accepted ICLR 2024 record
  (`Hf17y6u9BC`), not the earlier workshop record (`9eJv5PS27Q`).
- The pyvene entry now includes its complete official title, pages, DOI, and
  ACL Anthology record.
