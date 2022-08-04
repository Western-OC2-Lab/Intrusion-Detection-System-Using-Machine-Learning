# Intrusion-Detection-System-Using-Machine-Learning

This repository contains the code for the project "Intrusion Detection System Development for Autonomous/Connected Vehicles". The code and proposed Intrusion Detection System (IDSs) are general models that can be used in any IDS and anomaly detection applications. In this project, three papers have been published:  
* L. Yang, A. Moubayed, I. Hamieh and A. Shami, "[Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles](https://arxiv.org/pdf/1910.08635.pdf)," in 2019 IEEE Global Communications Conference (GLOBECOM), 2019, pp. 1-6, doi: 10.1109/GLOBECOM38437.2019.9013892.  
* L. Yang, A. Moubayed, and A. Shami, “[MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles](https://arxiv.org/pdf/2105.13289.pdf),” IEEE Internet of Things Journal, vol. 9, no. 1, pp. 616-632, Jan.1, 2022, doi: 10.1109/JIOT.2021.3084796.
* L. Yang, A. Shami, G. Stevens, and S. DeRusett, “LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles," in 2022 IEEE Global Communications Conference (GLOBECOM), 2022, pp. 1-6.

It proposed three **intrusion detection systems** by implementing many **machine learning** algorithms, including tree-based algorithms (**decision tree, random forest, XGBoost, etc.**), unsupervised learning algorithms (**k-means**), ensemble learning algorithms (**stacking**), and hyperparameter optimization techniques (**Bayesian optimization**)**.

- Another **intrusion detection system development code** using **convolutional neural networks (CNNs)** and **transfer learning** techniques can be found in: [Intrusion-Detection-System-Using-CNN-and-Transfer-Learning](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning)

- A comprehensive **hyperparameter optimization** tutorial code can be found in: [Hyperparameter-Optimization-of-Machine-Learning-Algorithms](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms)


## Implementation 
### Dataset 
CICIDS2017 dataset, a popular network traffic dataset for intrusion detection problems
* Publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html  
* For the purpose of displaying the experimental results in Jupyter Notebook, the sampled subsets of CICIDS2017 is used in the sample code. The subsets are in the "data" folder.

CAN-intrusion dataset, a benchmark network security dataset for intra-vehicle intrusion detection
* Publicly available at: https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset  
* Can be processed using the same code

### Code  
* [Tree-based_IDS_GlobeCom19.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/Tree-based_IDS_GlobeCom19.ipynb): code for the paper "Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles"  
* [MTH_IDS_IoTJ.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/MTH_IDS_IoTJ.ipynb): code for the paper "MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles"  

### Machine Learning Algorithms  
* Decision tree (DT)
* Random forest (RF)
* Extra trees (ET)
* XGBoost  
* Stacking
* K-means

### Hyperparameter Optimization Methods  
* Bayesian Optimization with Gaussian Processes (BO-GP)
* Bayesian Optimization with Tree-structured Parzen Estimator (BO-TPE)  

If you are interested in hyperparameter tuning of machine learning algorithms, please see the code in the following link:  
https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

### Requirements & Libraries  
* Python 3 
* [scikit-learn](https://scikit-learn.org/stable/)  
* [Xgboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [FCBF](https://github.com/SantiagoEG/FCBF_module)
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)  
* [hyperopt](https://github.com/hyperopt/hyperopt)   

## Contact-Info
Please feel free to contact us for any questions or cooperation opportunities. We will be happy to help.
* Email: [liyanghart@gmail.com](mailto:liyanghart@gmail.com)
* GitHub: [LiYangHart](https://github.com/LiYangHart) and [Western OC2 Lab](https://github.com/Western-OC2-Lab/)
* LinkedIn: [Li Yang](https://www.linkedin.com/in/li-yang-65a190176/)  
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
