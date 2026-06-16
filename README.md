# Intrusion-Detection-System-Using-Machine-Learning

This repository contains the code for the project "IDS-ML: Intrusion Detection System Development Using Machine Learning". The code and proposed Intrusion Detection System (IDSs) are general models that can be used in any IDS and anomaly detection applications. In this project, three papers have been published:  
* L. Yang, A. Moubayed, I. Hamieh and A. Shami, "[Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles](https://arxiv.org/pdf/1910.08635.pdf)," in 2019 IEEE Global Communications Conference (GLOBECOM), 2019, pp. 1-6, doi: 10.1109/GLOBECOM38437.2019.9013892.  
* L. Yang, A. Moubayed, and A. Shami, “[MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles](https://arxiv.org/pdf/2105.13289.pdf),” IEEE Internet of Things Journal, vol. 9, no. 1, pp. 616-632, Jan.1, 2022, doi: 10.1109/JIOT.2021.3084796.
* L. Yang, A. Shami, G. Stevens, and S. DeRusett, “[LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles](https://arxiv.org/pdf/2208.03399.pdf)," in 2022 IEEE Global Communications Conference (GLOBECOM), 2022, pp. 1-6, doi: 10.1109/GLOBECOM48099.2022.10001280.


The code introduction of this repository is publicly available at:  
* L. Yang, and A. Shami, “[IDS-ML: An open source code for Intrusion Detection System development using Machine Learning](https://www.sciencedirect.com/science/article/pii/S2665963822001300)," Software Impacts, vol. 14, pp. 1-4, 2022, doi: 10.1016/j.simpa.2022.100446.

This repository proposed three **intrusion detection systems** by implementing many **machine learning** algorithms, including tree-based algorithms (**decision tree, random forest, XGBoost, LightGBM, CatBoost etc.**), unsupervised learning algorithms (**k-means**), ensemble learning algorithms (**stacking, proposed LCCDE**), and hyperparameter optimization techniques (**Bayesian optimization**)**.

- Another **intrusion detection system development code** using **convolutional neural networks (CNNs)** and **transfer learning** techniques can be found in: [Intrusion-Detection-System-Using-CNN-and-Transfer-Learning](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning)

- A comprehensive **hyperparameter optimization** tutorial code can be found in: [Hyperparameter-Optimization-of-Machine-Learning-Algorithms](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms)


## Paper Abstract
### Paper 1:  Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles
&emsp; The use of autonomous vehicles (AVs) is a promising technology in Intelligent Transportation Systems (ITSs) to improve safety and driving efficiency. Vehicle-to-everything (V2X) technology enables communication among vehicles and other infrastructures. However, AVs and Internet of Vehicles (IoV) are vulnerable to different types of cyber-attacks such as denial of service, spoofing, and sniffing attacks. An intelligent IDS is proposed in this paper for network attack detection that can be applied to not only Controller Area Network (CAN) bus of AVs but also on general IoVs. The proposed IDS utilizes tree-based ML algorithms including decision tree (DT), random forest (RF), extra trees (ET), and Extreme Gradient Boosting (XGBoost). The results from the implementation of the proposed intrusion detection system on standard data sets indicate that the system has the ability to identify various cyber-attacks in the AV networks. Furthermore, the proposed ensemble learning and feature selection approaches enable the proposed system to achieve high detection rate and low computational cost simultaneously.

**<p align="center">Figure 1: The overview of the tree-based IDS model.</p>**
<p align="center">
<img src="https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/Figures/Tree-based_IDS_Overview.jpg" width="280" />
</p>

### Paper 2:  MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles
&emsp; Modern vehicles, including connected vehicles and autonomous vehicles, nowadays involve many electronic control units connected through intra-vehicle networks to implement various functionalities and perform actions. Modern vehicles are also connected to external networks through vehicle-to-everything technologies, enabling their communications with other vehicles, infrastructures, and smart devices. However, the improving functionality and connectivity of modern vehicles also increase their vulnerabilities to cyber-attacks targeting both intra-vehicle and external networks due to the large attack surfaces. To secure vehicular networks, many researchers have focused on developing intrusion detection systems (IDSs) that capitalize on machine learning methods to detect malicious cyber-attacks. In this paper, the vulnerabilities of intra-vehicle and external networks are discussed, and a multi-tiered hybrid IDS that incorporates a signature-based IDS and an anomaly-based IDS is proposed to detect both known and unknown attacks on vehicular networks. Experimental results illustrate that the proposed system can accurately detect various types of known attacks on the CAN-OTIDS-dataset representing the intra-vehicle network data and the CICIDS2017 dataset illustrating the external vehicular network data.  
&emsp; The proposed MTH-IDS framework consists of two traditional ML stages (data pre-processing and feature engineering) and four tiers of learning models: 
1. Four tree-based supervised learners — decision tree (DT), random forest (RF), extra trees (ET), and extreme gradient boosting (XGBoost) — used as multi-class classifiers for known attack detection; 
2. A stacking ensemble model and a Bayesian optimization with tree Parzen estimator (BO-TPE) method for supervised learner optimization; 
3. A cluster labeling (CL) k-means used as an unsupervised learner for zero-day attack detection; 
4. Two biased classifiers and a Bayesian optimization with Gaussian process (BO-GP) method for unsupervised learner optimization. 

**<p align="center">Figure 2: The overview of the MTH-IDS model.</p>**
<p align="center">
<img src="https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/Figures/MTH-IDS_Overview.png" width="700" />
</p>


### Paper 3:  LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles
&emsp; Modern vehicles, including autonomous vehicles and connected vehicles, have adopted an increasing variety of functionalities through connections and communications with other vehicles, smart devices, and infrastructures. However, the growing connectivity of the Internet of Vehicles (IoV) also increases the vulnerabilities to network attacks. To protect IoV systems against cyber threats, Intrusion Detection Systems (IDSs) that can identify malicious cyber-attacks have been developed using Machine Learning (ML) approaches. To accurately detect various types of attacks in IoV networks, we propose a novel ensemble IDS framework named Leader Class and Confidence Decision Ensemble (LCCDE). It is constructed by determining the best-performing ML model among three advanced ML algorithms (XGBoost, LightGBM, and CatBoost) for every class or type of attack. The class leader models with their prediction confidence values are then utilized to make accurate decisions regarding the detection of various types of cyber-attacks. Experiments on two public IoV security datasets (Car-Hacking and CICIDS2017 datasets) demonstrate the effectiveness of the proposed LCCDE for intrusion detection on both intra-vehicle and external networks. 

**<p align="center">Figure 3: The overview of the LCCCDE IDS model.</p>**
<p align="center">
<img src="https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/Figures/LCCDE_Overview.jpg" width="800" />
</p>


## Implementation 
### Dataset 
CICIDS2017 dataset, a popular network traffic dataset for intrusion detection problems
* Publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html  
* For the purpose of displaying the experimental results in Jupyter Notebook, the sampled subsets of CICIDS2017 is used in the sample code. The subsets are in the "data" folder.

CAN-OTIDS dataset (OTIDS benchmark for intra-vehicle intrusion detection; source: [CAN-intrusion-dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset))  
* Can be processed using the same code

### Code  
* [Tree-based_IDS_GlobeCom19.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/Tree-based_IDS_GlobeCom19.ipynb): code for the paper "Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles"  
* [MTH_IDS_IoTJ.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/MTH_IDS_IoTJ.ipynb): code for the paper "MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles"  
* [LCCDE_IDS_GlobeCom22.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/LCCDE_IDS_GlobeCom22.ipynb): code for the paper "LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles"  

#### MTH-IDS modular pipeline (Parquet + reports)

The package [mth_ids_pipeline](mth_ids_pipeline) reproduces the MTH-IDS paper (default `--protocol paper`), the published IoTJ notebook (`--protocol notebook`), or the **CAN-OTIDS** benchmark (`--protocol can`). Layout: `core/` (ML), `io/` (artifacts), `phases/` (executable steps), `orchestration/` (runner).

**Documentation (Portuguese):**

| Doc | Content |
|-----|---------|
| [docs/README.md](docs/README.md) | Index and quick start |
| [docs/ARQUITETURA.md](docs/ARQUITETURA.md) | 4 tiers, 3 experiments, package layout |
| [docs/EXECUCAO.md](docs/EXECUCAO.md) | Commands, folders, bootstrap, troubleshooting |
| [docs/PIPELINE_PHASES.md](docs/PIPELINE_PHASES.md) | Phases 1–13 + [manual CLI](docs/PIPELINE_PHASES.md#rodar-cada-fase-manualmente) |
| [docs/PROTOCOLO_CICIDS.md](docs/PROTOCOLO_CICIDS.md) | CICIDS2017 (Tables VII/IX/X) |
| [docs/PROTOCOLO_CAN.md](docs/PROTOCOLO_CAN.md) | CAN pipeline index (Tables VI/VIII/X) |
| [docs/PROTOCOLO_CAN_INTRUSION.md](docs/PROTOCOLO_CAN_INTRUSION.md) | Car-Hacking original (`merge_can --source original`) |
| [docs/PROTOCOLO_CAN_OTIDS.md](docs/PROTOCOLO_CAN_OTIDS.md) | OTIDS repack (`merge_can --source otids`) |
| [docs/PAPER_PROTOCOL.md](docs/PAPER_PROTOCOL.md) | Paper vs notebook preset comparison |

Prepare datasets:

**CICIDS2017:**
```bash
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
```

**CAN** (choose one data source — see [docs/PROTOCOLO_CAN.md](docs/PROTOCOLO_CAN.md)):
```bash
# Car-Hacking → pipeline_can_intrusion_*
python -m mth_ids_pipeline.utils.merge_can --source original
# OTIDS → pipeline_can_otids_* (pastas separadas)
python -m mth_ids_pipeline.utils.merge_can --source otids
```

**Fine: load + sampling, then LOAO (Table IX)** — run from the repo root. Step 1 can be skipped if `data/CICIDS2017_fine.csv` already exists. Step 2 forces regeneration of phases 1–2 (~27k rows; see [docs/EXECUCAO.md](docs/EXECUCAO.md#bootstrap-automático)). Step 3 auto-bootstraps `06_supervised_metrics.json` from merged Table VII if missing.

```powershell
# 1) Fine CSV (skip if data\CICIDS2017_fine.csv already exists)
python -m mth_ids_pipeline.utils.merge_cicids --profile fine

# 2) Regenerate load + k-means sampling (phases 1–2)
Remove-Item data\pipeline_mth_ids_fine\01_preprocessed.parquet -ErrorAction SilentlyContinue
Remove-Item data\pipeline_mth_ids_fine\02_sampled_kmeans.parquet -ErrorAction SilentlyContinue
python -m mth_ids_pipeline.run_all --label-profile fine --from 1 --to 2

# 3) LOAO (phases 7–12; may take many hours)
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao

# 4) Table IX report (after LOAO finishes)
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

Stop after step 2 for load + sampling only. If step 3 fails on `06_supervised_metrics.json`, run supervised first: `python -m mth_ids_pipeline.run_supervised --protocol paper`.

Single LOAO attack (e.g. Bot, label 1) — still re-runs phases 7–8 (~1 h) per attack:

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap `
  --attack-label 1
```

To resume phases 9–11 only (skip re-running phase 8), see [docs/PIPELINE_PHASES.md — Retomar LOAO](docs/PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais).

Supervised (paper — Table VII) → `data/pipeline_mth_ids_merged/`:
```bash
python -m mth_ids_pipeline.run_supervised --protocol paper
```

Anomaly LOAO (paper — Table IX) → `data/pipeline_mth_ids_fine/` (auto: fine 1–2 + merged Table VII → `06_…` copy):
```bash
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao
```

Compare metrics vs paper/notebook (writes to `results/`):

```bash
python -m mth_ids_pipeline.report_paper_tables --table all \
  --intermediate-dir data/pipeline_mth_ids_merged \
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

Output: `results/paper_comparison.json` and `results/tables_report.txt`.

**CAN** (Tables VI/VIII — `--protocol can`):

```powershell
python -m mth_ids_pipeline.utils.merge_can --source original
python -m mth_ids_pipeline.run_supervised --protocol can
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
# OTIDS: --protocol can_otids + pipeline_can_otids_*
```

See [docs/PROTOCOLO_CAN.md](docs/PROTOCOLO_CAN.md).

**Run each phase manually** (full commands, LOAO resume, flags): [docs/PIPELINE_PHASES.md — Rodar cada fase manualmente](docs/PIPELINE_PHASES.md#rodar-cada-fase-manualmente).

CLI quick reference:

| Phase | Module | `--intermediate-dir` | Typical extras |
| --- | --- | --- | --- |
| 1 | `phases.phase01_load_preprocess` | merged or fine | `--input data/CICIDS2017.csv` |
| 2 | `phases.phase02_sample_kmeans` | merged, fine, or `pipeline_can_otids_*` | `--frac 0.008` (CICIDS e CAN) + auto `--sample-all-classes` no CAN |
| 4 | `phases.phase04_feature_engineering` | merged / `pipeline_can_otids_merged` | `--fcbf-scope train`, `--optimize-ig` (split 80/20 inside) |
| 5 | `phases.phase05_smote` | merged only | skipped with `--protocol can` |
| 6 | `phases.phase06_supervised_models` | merged | `--cv-folds 10`, `--hpo-on-validation` |
| 7 | `phases.phase07_anomaly_datasets` | fine | `--work-dir …/loao/attack_N`, `--attack-label N` |
| 8 | `phases.phase08_anomaly_features` | fine | `--work-dir …/attack_N`, `--optimize-ig --optimize-kpca` |
| 9 | `phases.phase09_anomaly_cluster` | fine | `--work-dir …/attack_N` |
| 10 | `phases.phase10_anomaly_cluster_hpo` | fine | `--work-dir …/attack_N`, `--n-calls 15` or `--skip-hpo` |
| 11 | `phases.phase11_anomaly_biased` | fine | `--work-dir …/attack_N`, `--force-biased --optimize-p-star` |
| 12 | `phases.phase12_anomaly_loao` | fine | `--attack-label N`, orchestrates 7→11 |

### Machine Learning Algorithms  
* Decision tree (DT)
* Random forest (RF)
* Extra trees (ET)
* XGBoost  
* LightGBM  
* CatBoost  
* Stacking
* K-means

### Hyperparameter Optimization Methods  
* Bayesian Optimization with Gaussian Processes (BO-GP)
* Bayesian Optimization with Tree-structured Parzen Estimator (BO-TPE)  

If you are interested in hyperparameter tuning of machine learning algorithms, please see the code in the following link:  
https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

### Requirements & Libraries  

Install: `pip install -r requirements.txt` (Python 3.10+ recommended; tested with 3.13).

* Python 3.6+ 
* [scikit-learn](https://scikit-learn.org/stable/)  
* [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE (API without `n_jobs` in recent versions)
* [Xgboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [lightgbm](https://lightgbm.readthedocs.io/en/v3.3.2/Python-Intro.html)
* [catboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [FCBF](https://github.com/SantiagoEG/FCBF_module)
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)  
* [hyperopt](https://github.com/hyperopt/hyperopt)   
* [River](https://riverml.xyz/dev/)  

## Contact-Info
Please feel free to contact us for any questions or cooperation opportunities. We will be happy to help.
* Email: [liyanghart@gmail.com](mailto:liyanghart@gmail.com)
* GitHub: [LiYangHart](https://github.com/LiYangHart) and [Western OC2 Lab](https://github.com/Western-OC2-Lab/)
* LinkedIn: [Li Yang](https://www.linkedin.com/in/li-yang-phd-65a190176/)  
* Google Scholar: [Li Yang](https://scholar.google.com.eg/citations?user=XEfM7bIAAAAJ&hl=en) and [OC2 Lab](https://scholar.google.com.eg/citations?user=oiebNboAAAAJ&hl=en)

## Citation
If you find this repository useful in your research, please cite one of the following two articles as:  

L. Yang, A. Moubayed, I. Hamieh and A. Shami, "Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles," 2019 IEEE Global Communications Conference (GLOBECOM), 2019, pp. 1-6, doi: 10.1109/GLOBECOM38437.2019.9013892.  
```
@INPROCEEDINGS{9013892,
  author={Yang, Li and Moubayed, Abdallah and Hamieh, Ismail and Shami, Abdallah},
  booktitle={2019 IEEE Global Communications Conference (GLOBECOM)}, 
  title={Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles}, 
  year={2019},
  pages={1-6},
  doi={10.1109/GLOBECOM38437.2019.9013892}
  }
```

L. Yang, A. Moubayed, and A. Shami, “MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles,” IEEE Internet of Things Journal, vol. 9, no. 1, pp. 616-632, Jan.1, 2022, doi: 10.1109/JIOT.2021.3084796.
```
@ARTICLE{9443234,
  author={Yang, Li and Moubayed, Abdallah and Shami, Abdallah},
  journal={IEEE Internet of Things Journal}, 
  title={MTH-IDS: A Multitiered Hybrid Intrusion Detection System for Internet of Vehicles}, 
  year={2022},
  volume={9},
  number={1},
  pages={616-632},
  doi={10.1109/JIOT.2021.3084796}}
```

L. Yang, A. Shami, G. Stevens, and S. DeRusett, “LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles," in 2022 IEEE Global Communications Conference (GLOBECOM), 2022, pp. 1-6, doi: 10.1109/GLOBECOM48099.2022.10001280.
```
@INPROCEEDINGS{10001280,
  author={Yang, Li and Shami, Abdallah and Stevens, Gary and de Rusett, Stephen},
  booktitle={GLOBECOM 2022 - 2022 IEEE Global Communications Conference}, 
  title={LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles}, 
  year={2022},
  pages={3545-3550},
  doi={10.1109/GLOBECOM48099.2022.10001280}}
```
