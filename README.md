# Visa Certification Prediction and Candidate Profiling (Ensemble Learning)

## 📌 Business Problem Context
The U.S. Office of Foreign Labor Certification (OFLC) reviews hundreds of thousands of employer applications seeking to bring qualified foreign workers into the United States. To safeguard local market parity, the Immigration and Nationality Act (INA) mandates that certifications are only granted if there is an inadequate domestic talent supply and the foreign worker's compensation matches or exceeds regional prevailing wages. 

With case volume rising dramatically year over year, reviewing each application sequentially has become an operational strain.

**Objective:** Acting as a Data Scientist at the consulting firm **EasyVisa**, this project builds a robust **Ensemble Learning Classification Pipeline** to predict whether an application will be `Certified` or `Denied`. The solution facilitates candidate shortlisting and uncovers key socio-economic and educational drivers to optimize future application profiles.

---

## 📊 Dataset and Feature Dictionary
The model utilizes historical immigration metrics (`EasyVisa.csv`), which includes:
* **`case_status` (Target Variable):** Binary outcome indication (`Certified` or `Denied`).
* **`education_of_employee`:** The applicant's highest academic tier (High School, Bachelor's, Master's, Doctorate).
* **`has_job_experience`:** Indicator of prior relative professional experience (`Y` / `N`).
* **`prevailing_wage`:** The average geographic market salary for similarly employed workers in the target occupation.
* **`no_of_employees` & `yr_of_estab`:** Institutional scale indicators for the petitioning employer.
* **`continent` & `region_of_employment`:** Spatial indicators across international origins and local destinations.

---

## ⚙️ Modeling Journey and Ensemble Benchmarking
The analysis implemented a rigorous modeling pipeline, benchmarking standard base algorithms against powerful ensemble frameworks:
1. **Data Preprocessing:** Standardized skewed distribution features (`prevailing_wage`) and mapped categorical hierarchies via ordinal encoding and dummy generation.
2. **Overfitting Mitigation:** Early decision trees and bagging algorithms displayed clear overfitting tendencies (1.0 training accuracy vs. ~0.60 validation splits). 
3. **Boosting Implementation:** Utilized **AdaBoost (Adaptive Boosting)** and **Gradient Boosting Machines (GBM)** to systematically convert sequential weak learners into an optimal, generalized predictive boundary.

### **Final Model Performance Matrix**
The **Hyperparameter Tuned AdaBoost Classifier** was chosen as the operational champion due to its highly stable, generalized performance layout across all datasets:

| Dataset Cut | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 0.704 | 0.715 | 0.191 | 0.301 |
| **Validation Set** | 0.700 | 0.706 | 0.180 | 0.287 |
| **Operational Test Set** 🏆 | **0.700** | **0.711** | **0.175** | **0.280** |

*Note: The final model prioritized solid, generalized Precision (0.71) to ensure candidates flagged for fast tracked visa approval possess an incredibly high probability of true certification.*

---

## 💡 Data Driven Actionable Insights and Recommendations

Based on the feature importances extracted by the champion AdaBoost model, the following core criteria heavily dictate visa acceptance rates:

1. **The Academic Shield:** Higher education levels (`Master's` and `Doctorate`) exhibit the strongest statistical importance for securing a visa certification. The OFLC could introduce an upstream automated verification gate requiring proof of advanced degrees before formal application queuing.
2. **Experience Capitalization:** Candidates possessing direct prior job experience show an exponentially higher conversion rate than entry level applicants. Employers should actively structure international recruitment funnels around mid to senior profiles.
3. **Wage Standardization:** The unit and volume of `prevailing_wage` heavily dictate case statuses. Establishing strict, unified wage baselines across both domestic and foreign workers ensures complete INA statutory compliance while mitigating salary deflation risks.

---

## 🛠️ Tech Stack and Dependencies
* **Programming Environment:** Python
* **Machine Learning Library:** `scikit-learn` (`AdaBoostClassifier`, `GradientBoostingClassifier`, `GridSearchCV`)
* **Data Synthesis:** `pandas`, `numpy`
* **Plotting Toolkit:** `matplotlib`, `seaborn`

---

## 🤝 Connect with Me
* **LinkedIn:** www.linkedin.com/in/yash-rane-275867156
* **GitHub Main Portfolio:** https://github.com/yash-rane-17
