# Intrusion-Detection-System-Using-Machine-Learning

This repository contains the code for the project "Intrusion Detection System Development for Autonomous / Connected Vehicles". The code and proposed Intrusion Detection System (IDSs) are general models that can be used in any IDS and anomaly detection applications. In this project, three papers have been published:  
* L. Yang, A. Moubayed, I. Hamieh and A. Shami, "[Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles](https://arxiv.org/pdf/1910.08635.pdf)," in 2019 IEEE Global Communications Conference (GLOBECOM), 2019, pp. 1-6, doi: 10.1109/GLOBECOM38437.2019.9013892.  
* L. Yang, A. Moubayed, and A. Shami, “[MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles](https://arxiv.org/pdf/2105.13289.pdf),” IEEE Internet of Things Journal, vol. 9, no. 1, pp. 616-632, Jan.1, 2022, doi: 10.1109/JIOT.2021.3084796.
* L. Yang, A. Shami, G. Stevens, and S. DeRusett, “LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles," in 2022 IEEE Global Communications Conference (GLOBECOM), 2022, pp. 1-6.

It proposed three **intrusion detection systems** by implementing many **machine learning** algorithms, including tree-based algorithms (**decision tree, random forest, XGBoost, LightGBM, CatBoost etc.**), unsupervised learning algorithms (**k-means**), ensemble learning algorithms (**stacking, proposed LCCDE**), and hyperparameter optimization techniques (**Bayesian optimization**)**.

- Another **intrusion detection system development code** using **convolutional neural networks (CNNs)** and **transfer learning** techniques can be found in: [Intrusion-Detection-System-Using-CNN-and-Transfer-Learning](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning)

- A comprehensive **hyperparameter optimization** tutorial code can be found in: [Hyperparameter-Optimization-of-Machine-Learning-Algorithms](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms)


## Paper Abstract
### Paper 1:  Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles
&emsp; The use of autonomous vehicles (AVs) is a promising technology in Intelligent Transportation Systems (ITSs) to improve safety and driving efficiency. Vehicle-to-everything (V2X) technology enables communication among vehicles and other infrastructures. However, AVs and Internet of Vehicles (IoV) are vulnerable to different types of cyber-attacks such as denial of service, spoofing, and sniffing attacks. An intelligent IDS is proposed in this paper for network attack detection that can be applied to not only Controller Area Network (CAN) bus of AVs but also on general IoVs. The proposed IDS utilizes tree-based ML algorithms including decision tree (DT), random forest (RF), extra trees (ET), and Extreme Gradient Boosting (XGBoost). The results from the implementation of the proposed intrusion detection system on standard data sets indicate that the system has the ability to identify various cyber-attacks in the AV networks. Furthermore, the proposed ensemble learning and feature selection approaches enable the proposed system to achieve high detection rate and low computational cost simultaneously.
### Paper 2:  MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles
&emsp; Modern vehicles, including connected vehicles and autonomous vehicles, nowadays involve many electronic control units connected through intra-vehicle networks to implement various functionalities and perform actions. Modern vehicles are also connected to external networks through vehicle-to-everything technologies, enabling their communications with other vehicles, infrastructures, and smart devices. However, the improving functionality and connectivity of modern vehicles also increase their vulnerabilities to cyber-attacks targeting both intra-vehicle and external networks due to the large attack surfaces. To secure vehicular networks, many researchers have focused on developing intrusion detection systems (IDSs) that capitalize on machine learning methods to detect malicious cyber-attacks. In this paper, the vulnerabilities of intra-vehicle and external networks are discussed, and a multi-tiered hybrid IDS that incorporates a signature-based IDS and an anomaly-based IDS is proposed to detect both known and unknown attacks on vehicular networks. Experimental results illustrate that the proposed system can accurately detect various types of known attacks on the CAN-intrusion-dataset representing the intra-vehicle network data and the CICIDS2017 dataset illustrating the external vehicular network data.  
&emsp; The proposed MTH-IDS framework consists of two traditional ML stages (data pre-processing and feature engineering) and four tiers of learning models: 
1. Four tree-based supervised learners — decision tree (DT), random forest (RF), extra trees (ET), and extreme gradient boosting (XGBoost) — used as multi-class classifiers for known attack detection; 
2. A stacking ensemble model and a Bayesian optimization with tree Parzen estimator (BO-TPE) method for supervised learner optimization; 
3. A cluster labeling (CL) k-means used as an unsupervised learner for zero-day attack detection; 
4. Two biased classifiers and a Bayesian optimization with Gaussian process (BO-GP) method for unsupervised learner optimization. 

### Paper 3:  LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles
&emsp; Modern vehicles, including autonomous vehicles and connected vehicles, have adopted an increasing variety of functionalities through connections and communications with other vehicles, smart devices, and infrastructures. However, the growing connectivity of the Internet of Vehicles (IoV) also increases the vulnerabilities to network attacks. To protect IoV systems against cyber threats, Intrusion Detection Systems (IDSs) that can identify malicious cyber-attacks have been developed using Machine Learning (ML) approaches. To accurately detect various types of attacks in IoV networks, we propose a novel ensemble IDS framework named Leader Class and Confidence Decision Ensemble (LCCDE). It is constructed by determining the best-performing ML model among three advanced ML algorithms (XGBoost, LightGBM, and CatBoost) for every class or type of attack. The class leader models with their prediction confidence values are then utilized to make accurate decisions regarding the detection of various types of cyber-attacks. Experiments on two public IoV security datasets (Car-Hacking and CICIDS2017 datasets) demonstrate the effectiveness of the proposed LCCDE for intrusion detection on both intra-vehicle and external networks. 

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
* [LCCDE_IDS_GlobeCom22.ipynb](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/LCCDE_IDS_GlobeCom22.ipynb): code for the paper "LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles"  

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
* Python 3.6+ 
* [scikit-learn](https://scikit-learn.org/stable/)  
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

L. Yang, A. Shami, G. Stevens, and S. DeRusett, “LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles," in 2022 IEEE Global Communications Conference (GLOBECOM), 2022, pp. 1-6.
```
@INPROCEEDINGS{lyanggc22,
  author={Yang, Li and Shami, Abdallah and Stevens, Gary and DeRusett, Stephen},
  booktitle={2022 IEEE Global Communications Conference (GLOBECOM)}, 
  title={LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles}, 
  year={2022},
  pages={1-6},
  }
```
