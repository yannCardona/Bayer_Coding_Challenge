# Breast Cancer Data Dashboard

This was a Project I worked on in the context of a conding challenge from Bayer in summer 2023. 
The project is a **Dash** web application designed to analyze breast cancer data and perform predictions using a **Logistic Regression model**. The app provides visualizations of the data, feature analysis, and displays the performance of the logistic regression model in classifying breast cancer as **Benign** or **Malignant**. Users can input a radius measurement to get a prediction on whether the result is likely benign or malignant.

## Features

- **Data Table**: View the raw breast cancer dataset.
- **Correlation Matrix**: Analyze correlations between different features.
- **Feature Analysis**: Visualize data distributions, box plots, and violin plots for individual features.
- **Logistic Regression**: Train and evaluate a logistic regression model for cancer diagnosis classification.
- **Prediction**: Input radius value to predict whether the result is benign or malignant.

## Prerequisites

Before you begin, ensure you have the following installed on your macOS machine:

- Python 3.8 or higher
- pip (Python package installer)
- Virtualenv (optional but recommended)

## Installation

Follow the steps below to set up and run the app on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/yannCardona/Bayer_Coding_Challenge.git
cd Bayer_Coding_Challenge
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Set Up a Virtual Environment (Recommended)

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python dashboard.py
```
By default, the app will be accessible on http://127.0.0.1:8050/. Open your web browser and visit the URL to access the dashboard.
