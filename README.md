# Iris Dataset Classification Project

This project demonstrates the implementation of K-Nearest Neighbors (KNN) and Decision Tree algorithms to classify the Iris dataset. The project uses Python and popular libraries like `scikit-learn`, `pandas`, and `matplotlib` to train and evaluate the models. It also explores different feature combinations to understand the impact on classification accuracy.

## Project Overview

The Iris dataset is a classic dataset used for classification tasks, containing measurements of iris flowers from three species: Iris setosa, Iris versicolor, and Iris virginica. This project aims to compare the performance of KNN and Decision Tree algorithms on this dataset, examining the impact of various features and exploring their use in a real-world context.

## Installation

To run this project, ensure you have Python installed. You can use `pip` to install the required packages. Here's a list of dependencies:
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install seaborn
pip install joblib

Code Structure:
iris.csv: The Iris dataset used for training and testing.
model_training.ipynb: Jupyter notebook with code to train and evaluate the KNN and Decision Tree models.
knn_model.joblib, decision_tree_model.joblib: Saved KNN and Decision Tree models.
knn_model_petals.joblib, decision_tree_model_petals.joblib: Saved models trained using petal features.

Running the Code:
Load the Iris dataset and preprocess it (e.g., drop unnecessary columns).
Train the KNN and Decision Tree models on the dataset.
Evaluate the models on a test set and obtain accuracy scores.
Use pdb for debugging and joblib to save/load models.
Visualize the data with scatter plots and heatmaps to understand feature correlations.

Results:
KNN achieved an accuracy of 95.56% using all features and 97.78% using only petal features.
Decision Trees achieved similar results, with an accuracy of 95.56% using all features.
The combined approach (voting classifier) didn't yield a significant improvement.

Docker Containerization:
The project uses Docker to ensure consistent environments for running the code. To build the Docker image, create a Dockerfile and specify the required dependencies. Use the following commands to build and run the Docker container:
docker build -t iris_classification .
docker run -it iris_classification

GitHub Integration and Version Control:
The project uses GitHub for version control, with a commit history documenting the development process. Branching and merging strategies were used to manage feature development and bug fixes.

Additional Information
For more details on the Iris dataset, refer to the UCI Machine Learning Repository.

If you have any questions or suggestions, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more information.
