# Tomato Leaf Disease Classification Using Hybrid CNN–Vision Transformer Feature Fusion

## 1. Project Overview

This project investigates multiclass tomato leaf disease classification using a hybrid deep learning feature extraction and machine learning classification approach.

The main objective is to investigate whether combining visual representations extracted from Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) can improve classification performance compared with using either architecture family independently.

Instead of fine-tuning deep learning models, this approach uses pretrained models as **frozen feature extractors**. Their extracted feature vectors are combined through Feature Fusion and then classified using traditional machine learning algorithms.

The project evaluates three feature representation scenarios and four classifiers under a consistent experimental setup.

### Main Contributions

- Pretrained CNN and Vision Transformer feature extraction.
- Comparison of CNN-only, Transformer-only, and hybrid feature representations.
- Feature Fusion using Concatenation.
- Evaluation of four traditional machine learning classifiers.
- Five-fold Stratified Group Cross-Validation.
- Validation-based model selection.
- Independent final test evaluation.
- Quantitative and qualitative error analysis.

**Best Configuration:** Scenario 3 – Hybrid Feature Fusion + Logistic Regression.

**Final Test Accuracy:** 98.96%

**Final Test Macro F1-score:** 98.96%

---

## 2. Relationship to the Previous Project

This project is a methodological extension of my previous repository:

**[Tomato Leaf Disease Classification Using CNN Models](https://github.com/7alasaleh/Tomato-Leaf-Disease-Classification-Using-CNN-Models)**

Both projects use the same PlantVillage tomato leaf dataset, the same six selected classes, and the same dataset split.

However, they investigate different approaches to image classification.

| Aspect | Previous Project | Current Project |
|---|---|---|
| Main approach | End-to-end CNN classification | Deep Feature Extraction + ML |
| Deep learning models | Five CNN architectures | Two CNNs + One Vision Transformer |
| Model initialization | Trained from scratch | Pretrained ImageNet weights |
| Feature extraction | Learned jointly with classification | Frozen feature extractors |
| Feature fusion | Soft Voting of class probabilities | Concatenation of feature vectors |
| Classifiers | CNN classification heads and ensemble methods | SVM, KNN, Random Forest, Logistic Regression |
| Evaluation | Individual models and ensemble | Three feature scenarios and four classifiers |

### Previous Project

The previous project trained five CNN architectures from scratch:

- AlexNet
- ResNet18
- MobileNetV2
- EfficientNetB0
- DenseNet121

It also investigated ensemble learning using Soft Voting and Weighted Voting approaches.

The best individual model was EfficientNetB0, while the Soft Voting ensemble achieved:

- Accuracy: 99.63%
- Macro F1-score: 99.64%

### Current Project

The current project investigates a different question:

**Can combining pretrained CNN and Transformer representations through feature-level fusion provide an effective alternative to end-to-end CNN classification?**

It explores complementary visual representations from convolutional architectures and attention-based architectures, followed by traditional machine learning classification.

The two projects therefore provide complementary experimental perspectives on the same classification problem.

Their results should not be interpreted as a direct architectural ranking because their training strategies and classification pipelines are different.

---

## 3. Dataset

### Source

The dataset was obtained from Kaggle:

[New Plant Diseases Dataset – PlantVillage](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

Dataset directory used in the Kaggle environment:

`/kaggle/input/datasets/vipoooool/new-plant-diseases-dataset`

### Selected Classes

Six tomato leaf categories were selected:

1. `Tomato___Bacterial_spot`
2. `Tomato___Early_blight`
3. `Tomato___Late_blight`
4. `Tomato___Leaf_Mold`
5. `Tomato___Septoria_leaf_spot`
6. `Tomato___healthy`

Total images: **11,026**

### Dataset Splitting

The experiment used a previously established, group-aware dataset split. The original split was preserved without randomly splitting the images again.

| Subset | Images | Percentage |
|---|---:|---:|
| Training | 7,727 | 70.1% |
| Validation | 1,668 | 15.1% |
| Test | 1,631 | 14.8% |
| **Total** | **11,026** | **100%** |

The split manifest was stored separately:

`split_manifest.csv`

It contains image paths, class labels, group identifiers, leakage groups, and split assignments.

### Leakage Prevention

Since the source dataset contains augmented images, related images may share visual characteristics.

A group-aware splitting strategy was used to reduce the risk of information leakage.

The experiment used:

- Existing image group identifiers.
- SHA-256-based duplicate information.
- Combined leakage groups.

Before cross-validation, the group assignments were checked to ensure that no leakage groups were shared between the training and validation folds.

The same principle was maintained across the original Training, Validation, and Test subsets.

This approach reduces the risk of leakage from identified related images, although it cannot guarantee the complete elimination of all perceptually similar images.

---

## 4. Pretrained Feature Extraction Models

Three pretrained deep learning architectures were used as frozen feature extractors.

All models were initialized using the official pretrained ImageNet weights available through Torchvision.

### 4.1 ResNet18

- Architecture: Convolutional Neural Network.
- Feature dimension: 512.
- Original classification head: Removed.
- Role: Extraction of convolutional visual representations.

### 4.2 EfficientNetB0

- Architecture: Convolutional Neural Network.
- Feature dimension: 1,280.
- Original classification head: Removed.
- Role: Extraction of complementary visual representations using an efficient CNN architecture.

### 4.3 Vision Transformer B/16 (ViT-B/16)

- Architecture: Vision Transformer.
- Feature dimension: 768.
- Original classification head: Removed.
- Role: Extraction of transformer-based visual representations using image patches and self-attention.

### Feature Extractor Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Pretrained weights | ImageNet |
| Input image size | 224 × 224 |
| Feature extraction | Frozen |
| Gradient computation | Disabled |
| Device | NVIDIA Tesla T4 |
| Random seed | 42 |

All model parameters were frozen using `requires_grad=False`, and the models were kept in evaluation mode.

No fine-tuning or end-to-end neural network training was performed in this project.

### Preprocessing

The official preprocessing transformations associated with each pretrained model were used.

These transformations included:

- Image resizing.
- Center cropping to 224 × 224.
- Model-specific interpolation.
- ImageNet normalization.

The official preprocessing configuration was retained for each architecture rather than forcing identical interpolation settings across all models.

---

## 5. Feature Extraction

Features were extracted from all 11,026 images.

The final feature dimensions were:

| Model | Feature Dimension | Extracted Feature Matrix |
|---|---:|---|
| ResNet18 | 512 | 11,026 × 512 |
| EfficientNetB0 | 1,280 | 11,026 × 1,280 |
| ViT-B/16 | 768 | 11,026 × 768 |

The extracted feature matrices were saved as NumPy files to avoid repeating the feature extraction process during subsequent experiments.

Feature extraction was performed using PyTorch inference mode and GPU acceleration.

The extracted features were subsequently combined to construct three experimental scenarios.

---

## 6. Feature Fusion Scenarios

Three feature representation scenarios were designed.

### Scenario 1: CNN Feature Fusion

Combines ResNet18 and EfficientNetB0 features.

- ResNet18: 512 features.
- EfficientNetB0: 1,280 features.
- Fusion method: Concatenation.

**Total feature dimension: 1,792**

### Scenario 2: Vision Transformer Only

Uses ViT-B/16 features without CNN feature fusion.

- ViT-B/16: 768 features.

**Total feature dimension: 768**

### Scenario 3: Hybrid CNN–Transformer Feature Fusion

Combines all three feature extractors:

- ResNet18: 512 features.
- EfficientNetB0: 1,280 features.
- ViT-B/16: 768 features.

Fusion method: Concatenation.

**Total feature dimension: 2,560**

### Scenario Summary

| Scenario | Feature Extractors | Fusion Method | Feature Dimension |
|---|---|---|---:|
| Scenario 1 | ResNet18 + EfficientNetB0 | Concatenation | 1,792 |
| Scenario 2 | ViT-B/16 | None | 768 |
| Scenario 3 | ResNet18 + EfficientNetB0 + ViT-B/16 | Concatenation | 2,560 |

The purpose of these scenarios was to compare convolutional, transformer-based, and hybrid feature representations under the same classification framework.

---

## 7. Machine Learning Classifiers

Four traditional machine learning classifiers were evaluated on each feature representation scenario.

### 7.1 Support Vector Machine (SVM)

- Kernel: RBF.
- C: 1.0.
- StandardScaler applied through a Pipeline.

### 7.2 K-Nearest Neighbors (KNN)

- Number of neighbors: 5.
- Weighting: Distance.
- Distance metric: Euclidean.
- StandardScaler applied through a Pipeline.

### 7.3 Random Forest

- Number of estimators: 300.
- Maximum features: Square root.
- Random state: 42.
- No feature scaling.

### 7.4 Logistic Regression

- Maximum iterations: 2,000.
- Random state: 42.
- StandardScaler applied through a Pipeline.

Scaling was performed inside the classification Pipeline to prevent data leakage during cross-validation.

### Total Experiments

Three feature scenarios were evaluated using four classifiers:

**3 scenarios × 4 classifiers = 12 configurations.**

---

## 8. Cross-Validation Strategy

A five-fold Stratified Group Cross-Validation strategy was applied using:

`StratifiedGroupKFold`

Configuration:

| Parameter | Value |
|---|---|
| Number of folds | 5 |
| Shuffle | True |
| Random state | 42 |
| Input subset | Training only |
| Training samples | 7,727 |
| Group variable | `leakage_group` |

The grouping strategy ensured that samples belonging to the same leakage group were not distributed across the training and validation portions of an individual fold.

All five folds were checked, and no overlapping leakage groups were detected.

Each of the 12 configurations was evaluated across five folds.

**Total fold evaluations: 60.**

### Cross-Validation Results

The following table reports the mean Macro F1-score across the five folds.

| Classifier | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---:|---:|---:|
| SVM | 96.65% | 97.34% | 97.89% |
| KNN | 90.88% | 88.90% | 92.17% |
| Random Forest | 91.37% | 89.14% | 92.37% |
| Logistic Regression | 96.73% | 97.57% | **98.18%** |

### Cross-Validation Observations

- Scenario 3 achieved the highest mean Macro F1-score with all four classifiers.
- Logistic Regression achieved the strongest overall performance.
- SVM was a competitive alternative, particularly with ViT features.
- KNN and Random Forest achieved lower scores than SVM and Logistic Regression.
- The hybrid representation provided the best results within the evaluated configurations.

The highest cross-validation result was obtained by:

**Scenario 3 + Logistic Regression: 98.18% Macro F1.**

---

## 9. Validation Evaluation

After cross-validation, each classifier was trained on the complete training subset and evaluated using the separate validation subset.

- Training samples: 7,727.
- Validation samples: 1,668.

### Validation Accuracy Results

| Classifier | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---:|---:|---:|
| SVM | 98.08% | 98.80% | 98.68% |
| Logistic Regression | 97.54% | 97.66% | **98.86%** |
| KNN | 93.65% | 90.05% | 94.60% |
| Random Forest | 92.63% | 90.35% | 93.05% |

### Best Validation Configurations

| Configuration | Accuracy | Macro F1 |
|---|---:|---:|
| Scenario 3 + Logistic Regression | **98.86%** | **98.88%** |
| Scenario 2 + SVM | 98.80% | 98.82% |
| Scenario 3 + SVM | 98.68% | 98.69% |
| Scenario 1 + SVM | 98.08% | 98.11% |

### Model Selection

Scenario 3 + Logistic Regression was selected as the final configuration because it achieved the highest validation Accuracy and Macro F1-score and also performed best during cross-validation.

Scenario 2 + SVM was a close competitor, but the hybrid configuration was selected before accessing the final test results.

---

## 10. Final Test Evaluation

After selecting the best configuration, the Training and Validation subsets were combined.

Final training set:

- 9,395 images.

Independent test set:

- 1,631 images.

The selected model was trained on the combined training data and evaluated once on the held-out test set.

### Final Model

**Scenario 3 – Hybrid CNN–Transformer Feature Fusion + Logistic Regression**

### Final Test Results

| Metric | Result |
|---|---:|
| Test samples | 1,631 |
| Accuracy | **98.9577%** |
| Macro Precision | 98.9644% |
| Macro Recall | 98.9617% |
| Macro F1-score | **98.9620%** |
| Weighted F1-score | 98.9582% |
| Correct predictions | 1,614 |
| Incorrect predictions | 17 |
| Error rate | 1.04% |
| Training time | 3.98 seconds |
| Prediction time | 0.087 seconds |

The model achieved approximately 98.96% Accuracy and Macro F1-score on the independent test subset.

### Per-Class Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Bacterial Spot | 99.61% | 100.00% | 99.81% | 256 |
| Early Blight | 97.97% | 98.97% | 98.47% | 292 |
| Late Blight | 98.55% | 97.83% | 98.19% | 277 |
| Leaf Mold | 99.65% | 99.30% | 99.47% | 285 |
| Septoria Leaf Spot | 98.02% | 98.41% | 98.21% | 251 |
| Healthy | 100.00% | 99.26% | 99.63% | 270 |
| **Macro Average** | **98.96%** | **98.96%** | **98.96%** | **1,631** |

### Test Result Interpretation

Several observations can be made:

- All six classes achieved an F1-score above 98%.
- Bacterial Spot achieved 100% Recall.
- Healthy achieved 100% Precision.
- Late Blight had the lowest Recall at 97.83%.
- The small difference between Macro and Weighted F1 indicates balanced performance across the selected classes.

The final evaluation demonstrates strong classification performance within the selected dataset and experimental split.

---

## 11. Error Analysis

The final model incorrectly classified 17 out of 1,631 test images.

The predicted labels, true labels, and prediction confidence values were saved for further investigation.

### Main Misclassification Patterns

| True Class | Predicted Class | Number of Errors |
|---|---|---:|
| Late Blight | Early Blight | 3 |
| Late Blight | Septoria Leaf Spot | 3 |
| Early Blight | Late Blight | 2 |
| Septoria Leaf Spot | Early Blight | 2 |
| Early Blight | Septoria Leaf Spot | 1 |
| Septoria Leaf Spot | Late Blight | 1 |
| Septoria Leaf Spot | Leaf Mold | 1 |
| Leaf Mold | Bacterial Spot | 1 |
| Leaf Mold | Early Blight | 1 |
| Healthy | Late Blight | 1 |
| Healthy | Septoria Leaf Spot | 1 |

The most frequent confusion involved Early Blight, Late Blight, and Septoria Leaf Spot.

### Visual Inspection of High-Confidence Errors

Six incorrectly classified test images with prediction confidence above 90% were selected for qualitative visual inspection.

The observed confidence values ranged from 93.09% to 99.96%.

The inspection revealed:

- Four of the six inspected cases involved confusion between Early Blight and Late Blight.
- Several leaves contained irregular brown necrotic regions, dark spots, yellowish discoloration, and damaged tissue.
- Two cases involved confusion with Septoria Leaf Spot.
- Some small brown spots and relatively lighter centers appeared visually similar to patterns associated with Septoria Leaf Spot.

These observations suggest that overlapping visual symptoms and ambiguous disease appearance may contribute to classification errors.

Importantly, the visual inspection did not establish sufficient evidence to conclude that the original dataset annotations were incorrect.

The findings should therefore be interpreted as a qualitative analysis of selected error cases, rather than proof of annotation problems or architectural limitations.

### Confidence and Classification Reliability

The presence of highly confident incorrect predictions demonstrates that a high prediction confidence score does not necessarily guarantee a correct classification.

The confidence values should not be interpreted as calibrated probabilities without a separate calibration evaluation.

---

## 12. Final Experimental Comparison

The best configuration maintained similar performance across the evaluation stages.

| Evaluation Stage | Macro F1-score |
|---|---:|
| 5-Fold Cross-Validation | 98.18% |
| Validation | 98.88% |
| Final Test | 98.96% |

The results indicate consistent performance within the selected dataset split.

However, these results do not establish equivalent generalization performance on independently collected field images.

---

## 13. Project Outputs

The experiment generated the following artifacts:

### Extracted Features

- `ResNet18_features.npy`
- `EfficientNetB0_features.npy`
- `ViT_B16_features.npy`
- `feature_metadata.csv`

### Feature Scenarios

- `Scenario_1_CNN_Fusion.npy`
- `Scenario_2_ViT_Only.npy`
- `Scenario_3_Hybrid_Fusion.npy`

### Evaluation Results

- `validation_scenario_1.csv`
- `validation_scenario_2.csv`
- `validation_scenario_3.csv`
- `final_test_metrics.csv`
- `final_test_classification_report.csv`
- `final_test_confusion_matrix.csv`
- `final_test_predictions.csv`

### Error Analysis

- `final_test_misclassified_images.csv`

### Final Model

- `final_hybrid_logistic_regression.joblib`

### Visualization

- `final_test_confusion_matrix.png`

These files were generated in the Kaggle working environment. Their availability in the GitHub repository depends on which artifacts are subsequently committed.

---

## 14. Reproducibility and Experimental Design

The experiment used:

- Random seed: 42.
- PyTorch and Torchvision.
- Pretrained ImageNet weights.
- NVIDIA Tesla T4 GPU.
- Frozen feature extraction.
- Saved feature matrices.
- A fixed dataset split manifest.
- Five-fold Stratified Group Cross-Validation.
- Validation-based model selection.
- A final independent test evaluation.

StandardScaler was applied within Scikit-learn Pipelines for the relevant classifiers to avoid fitting scaling parameters on validation or test samples.

The final test set was not used for hyperparameter tuning or model selection.

Exact reproducibility may depend on software versions, GPU operations, and hardware configuration.

---

## 15. Limitations

Despite the strong test results, several limitations should be considered:

1. **Dataset-specific evaluation:** The experiment uses a selected subset of PlantVillage rather than a broad collection of real-world agricultural images.

2. **No external dataset evaluation:** The final model has not yet been evaluated on an independent external dataset.

3. **Frozen feature extractors:** The pretrained models were not fine-tuned specifically for tomato disease classification.

4. **Limited architecture selection:** Only ResNet18, EfficientNetB0, and ViT-B/16 were investigated.

5. **Single random seed:** The reported experiments use seed 42. Repeated experiments with multiple seeds would provide stronger evidence about performance variability.

6. **Potential perceptual similarity:** Group-aware splitting reduces identified leakage risks but cannot guarantee the removal of every near-duplicate or perceptually similar image.

7. **Confidence calibration:** Prediction confidence was inspected, but no formal probability calibration experiment was performed.

8. **Qualitative error analysis:** Only six high-confidence errors were visually inspected in detail.

9. **Domain shift:** Differences in lighting, background, camera quality, leaf orientation, and disease severity may affect performance on real-world images.

This project is an academic image-classification experiment and is not a validated agricultural diagnostic system.

---

## 16. Future Work

Potential improvements and research directions include:

1. Fine-tuning pretrained CNN and Vision Transformer architectures.
2. Comparing additional CNN and Transformer architectures.
3. Evaluating alternative feature fusion strategies, such as weighted fusion and dimensionality reduction.
4. Investigating feature selection and PCA for the hybrid feature representation.
5. Conducting repeated experiments with multiple random seeds.
6. Evaluating performance on an independent external dataset.
7. Investigating confidence calibration and uncertainty estimation.
8. Performing more extensive visual analysis of misclassified images.
9. Comparing computational cost, memory usage, and inference efficiency.
10. Investigating robustness to real-world image variations.

---

## 17. Conclusion

This project investigated a hybrid deep feature extraction and machine learning approach for six-class tomato leaf disease classification.

Three pretrained models were used as frozen feature extractors: ResNet18, EfficientNetB0, and ViT-B/16.

Three feature representation scenarios were constructed and evaluated using four traditional machine learning classifiers.

Across the 12 configurations and 60 cross-validation evaluations, the hybrid CNN–Transformer representation combined with Logistic Regression achieved the strongest overall results.

The final model achieved:

- **98.96% Test Accuracy**
- **98.96% Test Macro F1-score**
- **98.96% Test Macro Precision**
- **98.96% Test Macro Recall**

Only 17 of the 1,631 test images were incorrectly classified.

The results support the effectiveness of hybrid feature representations within the evaluated experimental setup. They also demonstrate that pretrained deep learning representations can be combined with traditional machine learning classifiers to achieve strong performance without end-to-end fine-tuning.

Together with the previous CNN classification project, this work provides a complementary investigation of deep learning-based tomato leaf disease classification, comparing end-to-end CNN training with pretrained CNN–Transformer feature extraction and feature-level fusion.

---

## Author

**7alasaleh**

GitHub: [https://github.com/7alasaleh](https://github.com/7alasaleh)

Previous Project: [Tomato Leaf Disease Classification Using CNN Models](https://github.com/7alasaleh/Tomato-Leaf-Disease-Classification-Using-CNN-Models)