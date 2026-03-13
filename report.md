## Credit Card Fraud Detection – Project Report

### 0. Executive Summary (Non‑Technical)

This project builds and compares several machine learning models to **detect fraudulent credit card transactions** in a highly imbalanced dataset from Kaggle. Fraudulent transactions are extremely rare (~0.17%), so the project focuses on:

- Using models and metrics that **prioritize catching fraud** (high recall) while avoiding too many false alarms.
- Designing a **reusable ML pipeline** that goes from raw data → preprocessing → model training → evaluation → deployment.
- Providing **two levels of explanation**:
  - A high‑level story for non‑technical stakeholders (risk managers, operations).
  - Detailed model, metric, and pipeline descriptions for technical readers.

Key outcomes:

- A tuned **XGBoost model** achieves the best discrimination, with **PR‑AUC ≈ 0.83 on validation** and **≈ 0.86 in cross‑validation**, and strong recall for fraud at reasonable precision.
- A **Logistic Regression model on the top 15 features**, combined with SMOTE and F2‑based thresholding, reaches **fraud recall ≈ 0.86** with acceptable precision and is highly interpretable.
- Additional models (Random Forest, Gradient Boosting, Neural Networks, semi‑supervised methods) are thoroughly explored and compared.
- The final pipeline is designed to be **maintainable and reusable**, supporting both batch and real‑time fraud scoring, plus monitoring and retraining.

### 1. Problem Statement

The goal of this project is to build and refine machine learning models that **detect fraudulent credit card transactions** in a highly imbalanced setting, and to package these models inside a **reusable, production‑oriented ML pipeline**.

The work is based on the Kaggle `mlg-ulb/creditcardfraud` dataset and extends across several experimental notebooks:
- EDA and preprocessing (`ML.ipynb`, `ML Project.pdf`).
- Logistic Regression on top features (`LogisticRegression_Top15.ipynb`).
- XGBoost modelling and tuning (`XGBoost.ipynb`).
- Random Forest, Gradient Boosting, Neural Networks.
- Semi‑supervised anomaly detection (`Credit_Card_Fraud_Semi_Supervised.ipynb`).

The key constraints are:
- **Extreme class imbalance**: only ~0.17% of transactions are fraudulent.
- **Anonymized feature space**: most predictors (`V1`–`V28`) are PCA-like components.
- **High cost of false negatives**: missing fraud is more costly than occasional false positives, so recall on the fraud class is prioritized.

### 2. Data Description

- **Source**: Kaggle dataset `mlg-ulb/creditcardfraud`.
- **Shape**: 284,807 rows × 31 columns.
- **Target**: `Class` (0 = legitimate, 1 = fraud).
- **Features**:
  - `Time`: seconds elapsed between the first transaction and the current transaction.
  - `Amount`: transaction amount.
  - `V1`–`V28`: anonymized numerical components (PCA-like).
- **Data quality**:
  - No missing values (`isnull().sum().sum() == 0`).
  - All features numeric (`float64` for 30 predictors, `int64` for `Class`).
- **Imbalance**:
  - Legitimate: 284,315 (≈ 99.83%).
  - Fraud: 492 (≈ 0.17%).

This imbalance makes naïve accuracy misleading and motivates the use of precision–recall metrics and recall‑focused thresholds.

From a business perspective:
- **False negatives (missed fraud)** are expensive (chargebacks, reputation).
- **False positives (incorrectly flagged transactions)** are also costly (customer friction, manual review).
- The project therefore frames the task as: **“Maximize fraud recall, subject to acceptable precision and alarm volume”**, and explicitly tests multiple operating thresholds.

### 3. Exploratory Data Analysis (EDA)

EDA was performed primarily in `ML.ipynb` and related notebooks, with summary visuals saved as:
- `02_amount_distribution.png`: transformed `Amount` distribution.
- `03_time_analysis.png`: temporal patterns of fraud vs. non‑fraud.

Key observations:
- **Class distribution**:
  - Confirmed that frauds are extremely rare (~0.17%), reinforcing the need for imbalance‑aware methods.
- **Amount**:
  - Raw amounts span a wide range (0 to ~25,691).
  - Robust scaling of `Amount` produces a distribution centered near 0 with reduced impact of extreme values, improving numerical stability for linear models.
- **Time**:
  - Standardized `Time` is used to capture potential temporal patterns; some clustering of frauds over certain time windows is visible but not dominant.
- **Feature scales**:
  - `V1`–`V28` are already standardized-like, but `Amount` and `Time` required explicit scaling steps.

### 4. Preprocessing and Data Splits

Across the main modeling notebooks the following preprocessing pipeline is used:

- **Scaling**:
  - `Amount`: `RobustScaler` (resistant to outliers).
  - `Time`: `StandardScaler`.
- **Shuffling**:
  - The dataset is shuffled with a fixed random seed for reproducibility.
- **Train/Validation/Test split** (from `LogisticRegression_Top15.ipynb` and XGBoost notebook):
  - Initial split: 80% train, 20% temp (stratified by `Class`).
  - Temp split: 50% validation, 50% test (stratified by `Class`).
  - Final shapes (approx.):
    - Train: 227,845 × 30.
    - Validation: 28,481 × 30.
    - Test: 28,481 × 30.
- **Imbalance handling**:
  - For tree models (XGBoost): `scale_pos_weight` is used to upweight fraud cases.
  - For linear models (Logistic Regression): class weights, custom class weight ratios, and SMOTE oversampling are explored.

### 5. Feature Selection and Importance

Two main strategies for understanding feature importance and reducing dimensionality are used:

- **Random Forest (Logistic Regression Top‑15 notebook)**:
  - A Random Forest with `class_weight='balanced'` is trained on all features.
  - Top 15 most important features (by RF importance) are selected:
    - `['V14', 'V10', 'V12', 'V4', 'V17', 'V3', 'V11', 'V16', 'V2', 'V9', 'V21', 'V7', 'V19', 'V20', 'V18']`.
  - Subsequent Logistic Regression experiments are restricted to these 15 features to improve interpretability and reduce variance.

- **XGBoost feature importance (XGBoost notebook)**:
  - Tuned XGBoost model’s feature importances are used to derive top‑k subsets.
  - Example top 10 features:
    - `['V14', 'V12', 'V4', 'V10', 'V17', 'V20', 'V8', 'Amount', 'V7', 'V19']`.
  - Example top 15 features (very consistent with RF):
    - `['V14', 'V12', 'V4', 'V10', 'V17', 'V20', 'V8', 'Amount', 'V7', 'V19', 'V3', 'V13', 'V15', 'V11', 'V25']`.

These consistent importance rankings across RF and XGBoost support focusing modeling and interpretability efforts on a small subset of high‑signal components, especially around `V14`, `V12`, `V4`, `V10`, `V17`, and `Amount`.

### 6. Logistic Regression Experiments (Top 15 Features)

All experiments in `LogisticRegression_Top15.ipynb` use:
- Only the RF top‑15 features.
- Stratified train/val/test splits as described above.
- A common evaluation helper that:
  - Computes PR‑AUC.
  - Searches thresholds that maximize F1 and F2 for the fraud class.
  - Reports per‑class precision, recall, F1 at:
    - Default threshold 0.5.
    - F1‑optimal threshold.
    - F2‑optimal threshold (fraud‑recall‑focused).

Main configurations and findings (all evaluated on the common validation set with the helper):

- **1. Baseline LogisticRegression (L2, no class weights)**:
  - Trained on top‑15 features with default `C=1.0`, `penalty='l2'`.
  - Good overall discrimination: **PR‑AUC ≈ 0.77** on validation.
  - Threshold analysis:
    - At **default 0.5**: very conservative; many frauds missed, high precision but sub‑optimal recall.
    - At **F1‑optimal**: balanced precision/recall; fraud recall improves substantially with only a moderate increase in false positives.
    - At **F2‑optimal**:
      - Fraud recall ≈ **0.84**.
      - Fraud precision ≈ **0.77**.
      - Legit recall remains ≈ 1.0 (almost all normal transactions still correctly classified).
  - Interpretation: even without explicit imbalance handling, **moving away from the 0.5 threshold** is critical; most of the gains come from choosing the right operating point.

- **2. `class_weight='balanced'`**:
  - Reweights the loss to give the minority class higher influence during training.
  - Validation **PR‑AUC ≈ 0.69**, somewhat lower than the baseline, but:
    - At F2‑optimal threshold:
      - Fraud recall ≈ **0.82**.
      - Fraud precision ≈ **0.63**.
    - At F1‑optimal threshold:
      - Fraud recall around mid‑0.7s with precision ~0.75.
  - Stronger custom weighting (`2x` the balanced ratio) shifts even more towards recall:
    - Fraud recall can approach the mid‑0.8s.
    - However, precision drops further, indicating more false alarms.
  - Interpretation: **class weights are a blunt tool**—they can push the model to see more fraud but often hurt precision and PR‑AUC if pushed too far.

- **3. L1 penalty (Lasso) + balanced**:
  - Uses `penalty='l1'`, `solver='saga'` with `class_weight='balanced'`.
  - Produces sparse coefficients—some of the 15 features are driven effectively to zero, highlighting the most impactful variables.
  - Performance:
    - PR‑AUC ≈ **0.69**, nearly identical to L2 balanced.
    - F2‑optimal fraud recall again sits around **0.82** with fraud precision ≈ **0.63**.
  - Interpretation: **L1 gives a more compact, explainable LR model** without materially changing the trade‑off envelope relative to L2; a good choice if feature selection is a priority.

- **4. Varying regularization strength (`C`)**:
  - Experiments sweep `C` over `{0.01, 0.1, 1.0}` (with `class_weight='balanced'`).
  - Observations:
    - Very strong regularization (`C=0.01`) shrinks all coefficients; the model becomes smoother, slightly reducing overfitting to the majority class.
    - `C=0.1` and `C=1.0` differ mainly in how aggressive the decision boundary is; F2‑optimal fraud recall remains in the **0.81–0.82** band.
    - PR‑AUC varies only slightly across these `C` values; the data are informative enough that LR is fairly robust to moderate changes in regularization.
  - Interpretation: **regularization stabilizes coefficients but does not dramatically change performance** once top features are fixed; imbalance handling and thresholding matter more than exact `C`.

- **5. SMOTE + Logistic Regression**:
  - Pipelines: `SMOTE(sampling_strategy=0.2/0.3/0.5, k_neighbors=5) → LogisticRegression`.
  - Validation behaviour:
    - SMOTE 0.2 + LR:
      - PR‑AUC ≈ **0.74**.
      - F2‑optimal fraud recall ≈ **0.84**, precision ≈ **0.73–0.74**.
    - SMOTE 0.3 + LR:
      - PR‑AUC ≈ **0.73**.
      - F2‑optimal fraud recall ≈ **0.84**, precision ≈ **0.71**.
    - **SMOTE 0.5 + LR**:
      - PR‑AUC ≈ **0.71** (slightly lower due to more synthetic points and overlap).
      - **F2‑optimal fraud recall ≈ 0.86**, precision ≈ **0.60**.
  - Interpretation:
    - As the synthetic minority ratio increases, **recall continues to climb**, but precision and PR‑AUC flatten or drop.
    - SMOTE 0.2–0.3 **offer a very attractive balance**: they retain much of the recall gain, with better precision and overall PR‑AUC than heavy oversampling.

**Conclusion for linear models**:
- A **Logistic Regression on the top‑15 features, combined with moderate SMOTE and F2‑based threshold tuning**, provides an interpretable and robust baseline with high fraud recall (≈ 0.85) and acceptable precision.

### 7. XGBoost Experiments

The XGBoost notebook builds a more expressive tree‑based model optimized for the imbalanced setting.

#### 7.1 Base XGBoost model

- **Data setup**:
  - Full 30 features after scaling (including `Time` and `Amount`).
  - Stratified train/val/test splits, as above.
  - Imbalance handled through `scale_pos_weight` (ratio of negatives to positives).

- **Initial tuned XGBoost (before hyperparameter search)**:
  - Validation performance:
    - Confusion matrix shows high fraud recall (~0.82) with strong overall accuracy.
    - Validation PR‑AUC ≈ **0.81**.
    - Weighted overall F1 ≈ **0.999** (dominated by the majority class).

#### 7.2 Cross‑validation and Hyperparameter Tuning

- **Cross‑validation**:
  - 5‑fold cross‑validation on the training set with `average_precision` (PR‑AUC) as the scoring metric.
  - Mean cross‑validated PR‑AUC ≈ **0.84**.
  - Confirms stable discrimination under resampling.

- **RandomizedSearchCV hyperparameter tuning**:
  - Search space includes:
    - `n_estimators`: [100, 200, 300]
    - `max_depth`: [3, 4, 6, 8]
    - `learning_rate`: [0.01, 0.05, 0.1]
    - `subsample`: [0.8, 1.0]
    - `colsample_bytree`: [0.8, 1.0]
  - Best configuration (example from notebook):
    - `n_estimators = 100`
    - `max_depth = 8`
    - `learning_rate = 0.1`
    - `subsample = 0.8`
    - `colsample_bytree = 0.8`
    - (plus previously set `scale_pos_weight` and `eval_metric="logloss"`)
  - Best cross‑validated PR‑AUC: **≈ 0.86**.

- **Validation performance of tuned XGBoost**:
  - Validation PR‑AUC ≈ **0.83**.
  - When thresholds are tuned on validation (similar to LR helper), the model achieves strong recall/precision trade‑offs at various operating points.

#### 7.3 SMOTE vs. scale_pos_weight for XGBoost

- An additional experiment uses **SMOTE before XGBoost**:
  - SMOTE + XGBoost validation PR‑AUC ≈ **0.81**.
  - Slightly worse than the tuned **scale_pos_weight** approach (≈ 0.83 on validation).

**Conclusion for XGBoost**:
- A **tuned XGBoost with `scale_pos_weight` and carefully chosen depth/learning rate** is the strongest performer in terms of PR‑AUC (~0.83 val; ~0.86 cross‑val), and offers high fraud recall with good precision.

#### 7.4 Random Forest, Gradient Boosting, Neural Networks, and Semi‑Supervised Models

Beyond Logistic Regression and XGBoost, the project explores several other model families. This section summarizes the main **quantitative** and **qualitative** observations from those notebooks.

- **Random Forest (RandomForestExperiments.ipynb)**:
  - Experiments cover:
    - Baseline RF with default hyperparameters.
    - RF with `class_weight='balanced'`.
    - Tuned RF using RandomizedSearchCV on:
      - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
  - Performance:
    - Baseline RF (no tuning, default depth) yields validation **PR‑AUC ≈ 0.81**, with fraud recall in the high‑0.7s at F2‑style thresholds.
    - Adding `class_weight='balanced'` and slightly deeper trees raises PR‑AUC into the **0.81–0.83** band and improves fraud recall at the cost of more false positives.
    - The tuned RF (RandomizedSearchCV):
      - **Best cross‑validated PR‑AUC ≈ 0.86**.
      - Validation PR‑AUC typically **≈ 0.82–0.83**.
      - Fraud recall at recall‑focused thresholds is similar to XGBoost, but precision is slightly lower (RF tends to produce more borderline positives).
  - Observations:
    - RFs are robust and relatively easy to tune; they capture non‑linear interactions and handle the PCA‑like features well.
    - They provide stable feature importance rankings that agree with XGBoost (e.g., emphasis on `V14`, `V10`, `V12`, `V4`, `V17`).
    - Compared to XGBoost, RF tends to need larger ensembles for best performance and can be somewhat less flexible in fine‑grained calibration/threshold shaping.

- **Gradient Boosting (GradientBoostingExperiments.ipynb)**:
  - The notebook explores:
    - Learning‑rate grids (e.g., 0.01, 0.05, 0.1).
    - Different depths and numbers of estimators.
    - Variants with and without class‑balancing and/or subsampling.
  - Performance:
    - A strong baseline GBDT (moderate depth, standard learning rate) reaches **validation PR‑AUC ≈ 0.85**, with fraud recall ≈ 0.83–0.85 at F2‑type thresholds.
    - Other tuned variants cluster in the **0.84–0.85** PR‑AUC range; small changes in learning rate or depth tend to shift the precision/recall frontier without large swings in PR‑AUC.
    - Under‑tuned or overly constrained models (too shallow, too low learning rate) can collapse to **PR‑AUC ≈ 0.42**, failing to separate fraud from noise.
    - A final consolidated GBDT model typically sits just below the tuned XGBoost in terms of PR‑AUC and fraud recall at similar precision.
  - Observations:
    - Classic Gradient Boosting is **competitive** and benefits from the same imbalance‑aware metrics as XGBoost.
    - Its training is more sensitive to learning rate and tree depth; without tuning, it can overfit the majority class or underfit the minority.
    - XGBoost’s regularization and built‑in handling of sparse patterns make it slightly more robust and easier to push to the performance ceiling.

- **Neural Networks (NeuralNetwork_Experiments.ipynb)**:
  - The NN notebook systematically varies:
    - Number of layers (e.g., shallow 1–2 layer MLPs vs. deeper networks).
    - Hidden sizes, activations (ReLU / LeakyReLU), and dropout.
    - Class weights and learning‑rate schedules.
  - Performance:
    - A simple 1–2 layer MLP with modest hidden units and no class weighting typically yields **PR‑AUC ~0.70–0.71**; it underfits the complex boundary.
    - Adding more layers/units and dropout, and using class weights, lifts PR‑AUC into the **0.78–0.82** band for the best configurations:
      - Frauds are detected with recall in the low‑0.8s but precision is often lower than tree‑based models at equivalent recall.
    - Overly large or insufficiently regularized networks can overfit the majority class and see PR‑AUC drop back into the high‑0.6s.
    - The best NNs remain slightly behind tuned XGBoost/RF in both PR‑AUC and in the sharpness of the precision–recall curve.
  - Observations:
    - NNs are **more sensitive** to optimizer settings, batch size, and epoch counts; small changes can materially shift PR‑AUC and convergence behavior.
    - They can, in principle, exploit complex non‑linearities in the transformed PCA space, but the gains over well‑tuned tree ensembles are modest for this dataset.
    - Interpretability is limited compared to LR coefficients and tree‑based feature importances, which is a consideration for fraud analysts.

- **Semi‑Supervised Models (Credit_Card_Fraud_Semi_Supervised.ipynb)**:
  - Two main approaches are tested using **only “normal” patterns” as primary signal**:
    - **Gaussian Mixture Model (GMM)**:
      - Fits a mixture distribution on normal transactions and flags points with low likelihood as anomalies.
      - Using a sensible anomaly threshold, GMM reaches **PR‑AUC ≈ 0.746**.
      - However, at recall‑focused operating points, precision is very low (**≈ 0.14**), meaning many legitimate transactions are incorrectly treated as suspicious.
    - **Isolation Forest**:
      - Uses random partitioning to isolate rare points.
      - Even after tuning the anomaly threshold (e.g., using the 99th percentile of anomaly scores), Isolation Forest attains only **PR‑AUC ≈ 0.11**, with very poor precision and recall for fraud.
  - Observations:
    - GMM is able to capture some structure in the distribution of legitimate transactions and yields a **useful anomaly score**, but:
      - The fraud manifold is not a “clean” outlier region—many frauds lie close to normal points in the PCA space.
      - To get high recall, the anomaly threshold must be set very low, which sweeps in a large number of normal points (hence precision ≈ 0.14).
    - Isolation Forest assumes that anomalies are **rare and easily isolatable** via random splits:
      - In this dataset, frauds do not consistently occupy sparse, isolated pockets; they often share regions of feature space with legitimate traffic.
      - The resulting isolation scores have **poor separation** between classes, leading to the observed PR‑AUC ≈ 0.11.
    - Overall, semi‑supervised methods are **not competitive with supervised trees/NNs** once labeled fraud examples are available, but they remain valuable as:
      - Additional weak signals to ensemble with supervised scores (e.g., for ranking or human review).
      - Cold‑start detectors when labels are scarce or delayed, where any anomaly signal is better than a random guess.

#### 7.5 Why Semi‑Supervised Learning Struggles on This Problem

Semi‑supervised anomaly detection methods (like GMM and Isolation Forest) implicitly assume that:
- The **normal class forms a compact, well‑modeled region** in feature space.
- Anomalies (frauds) are **few and far from that region**, so they appear as clear low‑likelihood or easily isolated points.

For this credit card fraud dataset, those assumptions break down:
- The PCA‑like features (`V1`–`V28`) are designed to anonymize the raw variables and mix information; as a result:
  - Fraud patterns **overlap heavily** with normal transactions in the transformed space.
  - There are no simple “edges” where all frauds live far away from the bulk of normal points.
- Fraud is not only rare, but also **heterogeneous**:
  - Some frauds look almost indistinguishable from normal activity except in subtle combinations of components.
  - Others may be extreme in one dimension but not across the full feature vector.
- Because of this overlap:
  - **GMM** must set a very low likelihood threshold to capture enough frauds, which pulls in a large mass of normal points, killing precision.
  - **Isolation Forest** sees many legitimate points as just as isolated as fraud points (due to sparse high‑dimensional geometry), so anomaly scores fail to separate classes.
- In contrast, supervised models (RF, XGBoost, LR, NNs) **directly learn class boundaries** from labeled fraud examples:
  - They can exploit tiny, label‑driven differences in the joint distribution that unsupervised density or isolation methods cannot reliably discover.
  - They naturally focus on the specific regions where frauds cluster, even when those regions are not globally “anomalous.”

As a result, while semi‑supervised methods are conceptually appealing for fraud detection, in this project:
- They underperform strong supervised baselines in PR‑AUC, precision, and recall.
- Their best role is as a **complementary anomaly score** fed into a supervised ensemble, or as a **temporary solution when labels are missing or delayed**, rather than as the primary detector.


### 8. Model Comparison (Summary)

On the validation data:

- **Logistic Regression (Top‑15 + SMOTE ~0.5 + F2 threshold)**:
  - Fraud recall: ≈ 0.86.
  - Fraud precision: ≈ 0.60.
  - PR‑AUC: ≈ 0.71.
  - Strengths: simple, interpretable coefficients on a small feature set; fast to train and deploy.

- **Tuned XGBoost (full features or top‑k importance‑based subset)**:
  - PR‑AUC (cross‑val): ≈ 0.86.
  - PR‑AUC (validation): ≈ 0.83.
  - Fraud recall and precision both strong at tuned thresholds.
  - Strengths: higher overall discrimination, non‑linear patterns, automatic handling of feature interactions; best candidate as a production model.

An effective strategy is to:
- Use **XGBoost as the primary production model**.
- Maintain **Logistic Regression (Top‑15 + SMOTE) as a simpler, interpretable backup/baseline**, useful for audits and explanations.

For technical readers, sections 6–7 detail the exact configurations (hyperparameters, feature subsets, imbalance handling, thresholds). For non‑technical stakeholders, the key message is:

- There is a clear performance hierarchy (XGBoost ≈ RF/GBDT > LR/NN > semi‑supervised).
- We understand **why** each model behaves as it does (e.g., tree ensembles capture interactions; semi‑supervised struggles due to overlapping distributions).
- The final recommendation explicitly balances **fraud‑catching power** and **operational cost**.

### 9. Proposed End‑to‑End ML Pipeline Architecture

This section describes the architecture you can draw for the project. It is designed to align with the notebooks while being deployment‑ready.

#### 9.1 High‑level stages

1. **Data Ingestion**
   - Pull raw transactions from:
     - Kaggle CSV (offline experiments).
     - Production sources (e.g., message queue, transaction DB) for deployment.
   - Apply schema validation and basic sanity checks.

2. **Preprocessing & Feature Engineering**
   - Handle any missing or corrupted records (drop or impute).
   - Scale:
     - `Amount` with `RobustScaler`.
     - `Time` with `StandardScaler`.
   - (Optional) Derive additional simple features:
     - Log‑transformed amount.
     - Time‑of‑day / day‑of‑week bins if raw timestamps become available.
   - Persist the fitted scalers for use at inference time.

3. **Train/Validation/Test Split**
   - Perform a stratified split into train, validation, and test.
   - Keep the split definition/versioned so evaluation is reproducible.

4. **Imbalance Handling**
   - **Tree‑based branch (XGBoost)**:
     - Use `scale_pos_weight` and optionally limit `max_depth` and `min_child_weight` to avoid overfitting on the majority class.
   - **Linear branch (Logistic Regression)**:
     - Use SMOTE in a pipeline (`SMOTE → LR`).
     - Alternatively, use `class_weight='balanced'` or custom weight dictionary.

5. **Model Training**
   - **XGBoost training**:
     - Train base model with reasonable defaults.
     - Run `RandomizedSearchCV` over:
       - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.
     - Save the best estimator and its hyperparameters.
   - **Logistic Regression training**:
     - Train baseline and SMOTE‑enhanced variants on top 15 features.
     - Compare L2 vs. L1 penalty and different `C` values.

6. **Threshold Optimization & Evaluation**
   - Using the validation set:
     - Compute PR‑AUC for each model.
     - Generate precision–recall curves for the fraud class.
     - Identify thresholds that optimize:
       - F1 (balanced).
       - **F2 (recall‑heavy)** for fraud.
   - Collect metrics:
     - PR‑AUC.
     - Fraud recall/precision at selected thresholds.
     - Confusion matrices.
   - Select a **primary operating threshold** (e.g. F2‑optimal) and possibly a **secondary threshold** for different business modes (high‑precision vs. high‑recall).

7. **Model Packaging**
   - Serialize:
     - Preprocessing pipeline (scalers, SMOTE config).
     - Final XGBoost model.
     - Final Logistic Regression model (backup).
     - Thresholds and configuration metadata.
   - Bundle as:
     - A Python package or
     - A REST/gRPC microservice (e.g., FastAPI/Flask) for real‑time predictions.

8. **Serving & Integration**
   - Real‑time scoring:
     - New transaction → preprocessing pipeline → feature vector → XGBoost → fraud probability → thresholding → decision (`fraud` / `legit` / `review`).
   - Batch scoring:
     - Periodic re‑scoring of historical data for backfills and monitoring.
   - Integrations:
     - Downstream case management / alerting systems.
     - Dashboards for fraud analysts (fraud rates, precision/recall over time).

9. **Monitoring & Retraining Loop**
   - Track:
     - Data drift (distribution changes in `V` components, `Amount`, `Time`).
     - Performance drift (PR‑AUC, fraud recall/precision on recent labeled data).
     - Alarm rates vs. analyst capacity.
   - Define retraining triggers:
     - Time‑based (e.g. monthly/quarterly).
     - Performance‑based (e.g. fraud recall dropping below a threshold).
   - Retraining process:
     - Re‑ingest a recent window of data.
     - Refit scalers, re‑tune XGBoost and LR, re‑select thresholds.
     - Re‑deploy updated models after offline validation.

#### 9.2 Architecture diagram (text/mermaid sketch)

You can draw the following flow as your high‑level architecture:

```mermaid
flowchart LR
    A[Data Sources<br/>Kaggle CSV / Prod DB] --> B[Ingestion & Validation]
    B --> C[Preprocessing<br/>Scaling, Cleaning]
    C --> D[Train/Val/Test Split]
    D --> E1[Imbalance Handling<br/>scale_pos_weight (XGB)]
    D --> E2[Imbalance Handling<br/>SMOTE (LR)]
    E1 --> F1[XGBoost Training<br/>+ Hyperparameter Tuning]
    E2 --> F2[Logistic Regression Training<br/>Top-15 Features]
    F1 --> G[Threshold Tuning<br/>PR-AUC, F1/F2 on Val]
    F2 --> G
    G --> H[Model Packaging<br/>Pipelines + Thresholds]
    H --> I[Online & Batch Serving]
    I --> J[Monitoring & Feedback<br/>Drift, Performance]
    J --> B
```

This diagram corresponds closely to the notebooks:
- EDA and preprocessing (`ML.ipynb` and others).
- Logistic Regression experiments on top features.
- XGBoost training and hyperparameter tuning.

### 10. Recommendations and Next Steps (Technical + Non‑Technical Views)

- **Primary production model**: tuned XGBoost with `scale_pos_weight`, trained on the full or importance‑filtered feature set, using F2‑optimized thresholding for fraud recall.
- **Interpretability companion**: Logistic Regression on top‑15 features with SMOTE and F2 threshold; log/reg coefficients can be exposed to analysts.
- **Further work**:
  - Explore calibrated probabilities (Platt scaling or isotonic regression) to improve threshold stability.
  - Add more robust time‑based features if richer timestamps are available.
  - Evaluate semi‑supervised or anomaly‑detection models from `Credit_Card_Fraud_Semi_Supervised.ipynb` as complementary detectors.
  - Implement monitoring dashboards for live fraud precision/recall and drift detection.

From a **non‑technical** angle:
- We selected a **primary algorithm** (XGBoost) that best balances how many frauds it catches versus how many false alerts it raises.
- We keep a **simpler, explainable model** (Logistic Regression) alongside it so we can always tell a clear story about **which features drive fraud risk**.
- We built a pipeline that can be embedded into your systems (online and batch) and **monitored over time**, so the model continues to work as customer behaviour changes.

From a **technical** angle:
- The pipeline consists of modular, testable components (preprocessing, imbalance handling, training, threshold tuning, packaging, serving).
- Hyperparameter search and threshold optimization are driven by **PR‑AUC and F1/F2 on a held‑out validation set**, making the selection process explicit and reproducible.

### 11. Reflection on Process and Learning

This section reflects on how the problem was framed, how the modelling approach evolved, and what was learned.

- **Framing the problem**:
  - Initial framing used accuracy and ROC‑AUC, but early experiments revealed that high accuracy could be achieved by **predicting “not fraud” almost always**.
  - This led to a reframing around **precision–recall**, especially PR‑AUC and recall for the fraud class, and to the adoption of F2‑optimized thresholds.
  - The project thus moved from “generic classification” to “cost‑sensitive fraud detection with explicit recall targets.”

- **Model refinement and fine‑tuning**:
  - The journey started with **simple Logistic Regression**, then gradually added:
    - Feature selection (RF and XGBoost importances).
    - Imbalance strategies (class weights, SMOTE).
    - Threshold tuning based on F1/F2.
  - Tree ensembles were then introduced (RF, GBDT, XGBoost), followed by **systematic hyperparameter tuning** and cross‑validation.
  - Neural networks and semi‑supervised models were explored as alternatives, but empirical results and practical considerations (stability, interpretability) placed them behind tuned XGBoost and RF.

- **Evaluation and metrics**:
  - A major learning point was how different metrics can tell very different stories:
    - Accuracy and ROC‑AUC looked “great” even for models that missed many frauds.
    - PR‑AUC and class‑specific recall/precision exposed the real trade‑offs.
  - Building dedicated helpers (in the notebooks) to compute **PR‑AUC, confusion matrices, and F1/F2‑optimal thresholds** was essential for consistent, fair comparisons.

- **Pipeline thinking vs. one‑off modelling**:
  - Early work focused on individual notebooks. Over time, the need for:
    - Reusable preprocessing (same scalers for training and inference).
    - Clear train/val/test separation.
    - Configurable imbalance handling.
    - A central place for thresholds and model metadata.
  - This naturally led to the **pipeline architecture** in section 9, which can be translated into production code with limited friction.

- **Semi‑supervised lessons**:
  - Semi‑supervised anomaly detectors were initially attractive because fraud is rare and “anomalous.”
  - Empirical results showed that in this dataset, fraud **does not sit in a neat “outlier shell”** around normal behaviour; it overlaps heavily with legitimate transactions in PCA space.
  - This demonstrated that **having labeled fraud data, even if limited, is crucial**, and that supervised models can exploit subtle label‑driven differences that density or isolation methods miss.

- **What I would improve with more time**:
  - Formalize the code into a **Python package** with unit tests for each pipeline stage.
  - Add **model calibration** and explicit cost curves to link metrics directly to business KPIs (e.g., dollars saved vs. analyst hours).
  - Explore **stacked/ensembled models** that combine XGBoost, RF, and NN outputs, plus semi‑supervised scores, into a meta‑classifier.

Overall, the project demonstrates:
- A full cycle from **problem framing → modelling → evaluation → pipeline design → reflection**.
- A clear understanding of **why certain models and metrics are preferred for fraud detection**.
- The ability to translate notebook experiments into a **clean, reusable architecture**.

### 12. Repository and Reproducibility

- **Git repository**:
  - Main repo: `[YOUR_GITHUB_REPO_URL_HERE]`
  - All notebooks (`*.ipynb`), figures (`*.png`), and this `report.md` live in the same repo.
  - Scripts/notebooks are organized so that:
    - EDA and preprocessing can be rerun end‑to‑end.
    - Model training notebooks (Logistic Regression, XGBoost, RF/GBDT/NN, Semi‑Supervised) can be executed independently, using the same data loading and preprocessing steps.
- **Collaboration**:
  - Add `DialaE` as a collaborator with read access on the GitHub repo, so the full code and experiments are visible.
- **How to reproduce key results**:
  - Run the EDA/preprocessing notebook to:
    - Download/validate the Kaggle dataset.
    - Apply scaling and create the stratified train/val/test splits.
  - Run the modelling notebooks:
    - `LogisticRegression_Top15.ipynb` → linear baseline + SMOTE experiments.
    - `XGBoost.ipynb` → tuned XGBoost + feature importance.
    - `RandomForestExperiments.ipynb`, `GradientBoostingExperiments.ipynb`, `NeuralNetwork_Experiments.ipynb`, `Credit_Card_Fraud_Semi_Supervised.ipynb` → alternative models.
  - Compare metrics and thresholds as documented in this report.

Figures and tables (e.g., amount distribution, time analysis, feature importance plots, PR‑AUC comparison tables) should be **included selectively** in your final submitted document to:
- Illustrate class imbalance and feature distributions.
- Highlight the performance of the best models (XGBoost vs. LR vs. RF/GBDT).
- Support the narrative about why semi‑supervised methods are not chosen as primary detectors.

