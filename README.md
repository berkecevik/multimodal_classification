# Multimodal Classification Using Classical Machine Learning Models in ROS 2

This ROS 2 project applies classical machine learning classification algorithms on multiple real-world datasets. Each dataset is processed, modeled, and visualized using Python, scikit-learn, and ROS 2 Python nodes.

##  Project Structure

```
multimodal_classification_ros2/
├── fruit_classifier/                # ROS 2 Python package
│   ├── fruit_classifier/
│   │   ├── fruit_node.py            # Logistic Regression on Fruit Dataset
│   │   ├── iris_node.py             # KNN, Decision Tree, Random Forest on Iris Dataset
│   │   ├── svm_node.py              # Linear SVM for Cancer and Penguin Datasets
│   │   ├── fruit_processed.csv
│   │   ├── iris_processed.csv
│   │   ├── cancer_processed.csv
│   │   └── penguin_processed.csv
│   ├── package.xml
│   └── setup.py
├── plots/
    ├── fruit_plot.png
    ├── iris_plot.png
    ├── breast_cancer_plot.png
    └── penguin_plot.png
```

## Classification Tasks and Models

### 1. Fruit Dataset – Logistic Regression
- Objective: Classify apples vs oranges using weight, sphericity, and color
- Model: Logistic Regression


### 2. Iris Dataset – KNN, Decision Tree, Random Forest
- Objective: Classify three iris flower species
- Models: K-Nearest Neighbors, Decision Tree, Random Forest

### 3. Breast Cancer & 🐧 Penguin Dataset – Linear SVM
- Breast Cancer: Classify malignant vs benign tumors
- Penguins: Multiclass classification of three penguin species
- Model: Support Vector Machine (Linear)


## Author

- **[Berke Cevik]**
- [Master of Software Engineering], [University of Europe]
