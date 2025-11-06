# B3_GRP6_ML_MINI_PROJECT

## Overview
This repository hosts the implementation of the Machine Learning Mini Project for Group B3 at K.J. Somaiya School of Engineering. It reproduces and extends the methodology from the 2022 arXiv paper:  
The Severity Prediction of the Binary and Multi-Class Cardiovascular Disease: A Machine Learning-Based Fusion Approach"
by Hafsa Binte Kibria and Abdul Matin (arXiv:2203.04921v1).  

[Original Paper PDF](B3_ML_GROUP_6_THE_SEVERITY_PREDICTION_OF_THE_BINARY_AND_MULTI_CLASS_CARDIOVASCULAR_DISEASE.pdf)  

The paper develops weighted score fusion models integrating six ML algorithms—Artificial Neural Network (ANN), Support Vector Machine (SVM), Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), and AdaBoost—for predicting Cardiovascular Disease (CVD): binary classification (presence/absence) and multi-class severity (Low/Medium/High risk). The fusion combines probabilities from model pairs using \( P_{fusion} = p \cdot P_{model1} + (1-p) \cdot P_{model2} \), optimizing \( p \) via grid search to maximize accuracy. This yields superior performance (up to 75% multi-class and 95% binary accuracy on the original dataset) by exploiting complementary model behaviors.  

**Project Compliance with Guidelines**:  
- **Paper**: Healthcare-focused (CVD diagnosis/risk stratification); from engineering context (2022).  
- **Reproduction**: From-scratch implementation in Python (Scikit-learn, Keras/TensorFlow), drawing from authors' GitHub: [Weighted Score Fusion](https://github.com/hafsakibria/Weightedscorefusion).  
- **Extension**: Ported to a new Kaggle dataset (70k records; distinct from original). Added feature engineering, oversampling, and detailed multi-class evaluation.  
- **Real-World Relevance**: Tackles CVD (leading global killer: ~18M deaths/year per WHO); supports early intervention and personalized risk assessment.  

[Project Poster PDF](16014224811_ML_MINI_PROJECT_Poster_B3_RISHI_MANE.pdf) – Features architecture diagrams, results tables, ROC curves, and insights.  

## Methodology
### Core Components
1. Data Preprocessing:  
   - Load CSV; engineer features: BMI = weight / (height/100)^2, BP_diff = ap_hi - ap_lo, age_years = age / 365.25.  
   - Outlier removal (e.g., height <100cm or >250cm); scaling (StandardScaler for ANN/SVM/LR).  
   - Targets: Binary (cardio: 0=No, 1=Yes); Multi-class (0=Low, 1=Medium, 2=High risk via thresholds on age/BP/cholesterol/gluc).  
   - Imbalance Handling: RandomOverSampler (imbalanced-learn) for multi-class.  
   - Split: Stratified 70-30 train-test.  

2. Individual Models:  
   | Model     | Key Hyperparameters                          |  
   |-----------|----------------------------------------------|  
   | ANN  | 2-3 hidden layers, ReLU, L2 reg (0.001), Adam/SGD optimizer, early stopping. |  
   | RF   | n_estimators=200, class_weight='balanced'.   |  
   | DT   | max_depth=6-8, min_samples_leaf=5.           |  
   | AdaBoost| n_estimators=100-200, learning_rate=0.05. |  
   | SVM  | kernel='rbf', C=1.0, probability=True.       |  
   | LR*  | solver='lbfgs', max_iter=500-1000, multi_class='multinomial' (multi). |  

3. Fusion Approach:  
   - Pairs: RF+ANN, DT+AdaBoost, SVM+LR (binary/multi-class variants).  
   - Optimization: Grid search \( p \in [0,1] \) (step=0.05) to max accuracy on fused probs.  
   - Decision: >0.5 threshold (binary); argmax (multi-class).  
   - Evaluation: Accuracy, F1-score, AUC-ROC (binary); Macro AUC, confusion matrix (multi-class). Visuals: ROC curves, bar plots (matplotlib/seaborn).  

### Workflow Diagram (from Poster)
```
Patient Input (Age, Height, Weight, BP, Cholesterol, etc.)
↓
Preprocessing (Scaling, BMI/BP_diff, Oversampling)
↓
Model 1 (e.g., RF) → Probabilities P1    Model 2 (e.g., ANN) → Probabilities P2
↓                                                ↓
Weighted Fusion: P_fusion = p * P1 + (1-p) * P2  (p optimized)
↓
Threshold/Argmax → Output: CVD Binary (0/1) or Severity (Low/Med/High)
```

## Dataset
- Source: [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (~70,000 anonymized records from Russian patients).  
- Features (11): id, age (days), gender (1/2), height (cm), weight (kg), ap_hi/lo (BP mmHg), cholesterol (1-3), gluc (1-3), smoke/alco/active (0/1).  
- Repo Files:  
  - `cardio_train.csv`: Raw binary-labeled data.  
  - `binary_classification.csv`: Processed binary features/target.  
  - `multiclassification.csv`: Engineered multi-class version.  
- Notes: Multi-class derived from risk combos (e.g., high age+BP=High risk). Pre-sampling distribution: ~50% CVD positive (binary); imbalanced multi-class balanced post-oversampling.  

## Results
Tested on Kaggle extension. Fusions excel by 3-5% over singles via diversity (e.g., RF for interactions, ANN for non-linearity). See poster for full plots/confusion matrices.  

### Binary Classification
| Fusion Model   | Accuracy | F1-Score | AUC-ROC |
|----------------|----------|----------|---------|
| RF + ANN      | 0.7289  | 0.7156  | 0.8045 |
| DT + AdaBoost | 0.7254  | 0.7198  | 0.8012 |
| SVM + LR      | 0.7243  | 0.7089  | 0.7998 |

### Multi-Class Classification
| Fusion Model   | Accuracy | Macro AUC |
|----------------|----------|-----------|
| ANN + RF      | 0.7845  | 0.8723   |
| DT + AdaBoost | 0.7621  | 0.8456   |
| SVM + LR      | 0.7534  | 0.8389   |

Optimal Weights (examples):  
- Binary RF+ANN: p=0.65 (RF).  
- Multi-class ANN+RF: p=0.60 (ANN).  

Insights (Poster Highlights):  
- Ensemble diversity key: Tree + neural/linear fusions best.  
- Oversampling +8% multi-class acc; BMI/BP_diff boost +2%.  
- Top Features: Age > BMI > systolic BP > cholesterol.  
- Lifestyle (smoke/alco) weaker (bias?); gender-specific importance variance.  
- Fixed 50-50 fusion: 2-3% worse than optimized.  

## Setup & Execution
1. Clone:  
   ```
   git clone https://github.com/rishimane-svg/B3_GRP6_ML_MINI_PROJECT.git
   cd B3_GRP6_ML_MINI_PROJECT
   ```

2. Dependencies:  
   ```
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Data: CSVs already included; if missing, download `cardio_train.csv` from Kaggle.  

4. Run:  
   - Start Jupyter: `jupyter notebook`.  
   - **Binary**: `FUSION_BIN(1).ipynb` (main fusion); or `ANN+RF_BINARY_CLASSIFICATION.ipynb`, `DECISION_TREE+ADABOOST_BINARY_CLASSIFICATION.ipynb`, `SVM+LR_BINARY_CLASSIFICATION.ipynb`.  
   - **Multi-Class**: `FUSION_MULTI(1).ipynb` (main, w/ oversampling); or `ANN+LR_MULTI.ipynb`, `LR+RF_MULTI.ipynb`, `SVM+ANN_MULTI.ipynb`.  
   - Outputs: Console metrics, plots saved inline. ~5-10 min runtime (CPU; GPU accelerates ANN).  

## Repository Structure
```
B3_GRP6_ML_MINI_PROJECT/
├── FUSION_BIN(1).ipynb                          # Binary fusion (all pairs)
├── FUSION_MULTI(1).ipynb                        # Multi-class fusion (all pairs, oversampling)
├── ANN+RF_BINARY_CLASSIFICATION.ipynb           # Binary RF+ANN details
├── DECISION_TREE+ADABOOST_BINARY_CLASSIFICATION.ipynb  # Binary DT+AdaBoost
├── SVM+LR_BINARY_CLASSIFICATION.ipynb           # Binary SVM+LR
├── ANN+LR_MULTI.ipynb                           # Multi-class ANN+LR
├── LR+RF_MULTI.ipynb                            # Multi-class LR+RF
├── SVM+ANN_MULTI.ipynb                          # Multi-class SVM+ANN
├── cardio_train.csv                             # Raw dataset
├── binary_classification.csv                    # Processed binary data
├── multiclassification.csv                      # Processed multi-class data
├── 16014224811_ML_MINI_PROJECT_Poster_B3_RISHI_MANE.pdf  # Project poster
├── B3_ML_GROUP_6_THE_SEVERITY_PREDICTION_OF_THE_BINARY_AND_MULTI_CLASS_CARDIOVASCULAR_DISEASE.pdf  # Original paper
└── requirements.txt                             # Python dependencies
```

## Limitations & Future Work
- Challenges: Cross-sectional data limits causality; oversampling risks minority overfitting; needs robust probability calibration.  
- Extensions (Poster):  
  - Stacking ensembles with meta-learners.  
  - Temporal/longitudinal data integration.  
  - Interpretability (SHAP/LIME) for clinical adoption.  
  - Web-based deployment for real-time screening.  
  - Continuous risk regression over discrete classes.  

## References
1. Kibria, H. B., & Matin, A. (2022). arXiv:2203.04921.  
2. Dataset: Kaggle CVD Dataset.  
3. Original Repo: https://github.com/hafsakibria/Weightedscorefusion.  


Author: Rishi Mane (Roll No: 16014224811, IT Department)  
Institution: K.J. Somaiya School of Engineering  
Contact: rishi.mane@somaiya.edu | [Somaiya.edu](https://somaiya.edu/)
