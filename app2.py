import subprocess
import sys

# 需要的依赖列表
required_packages = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "scipy",
    "Pillow",
    "openpyxl",  # 处理Excel文件
    "streamlit"
]

# 安装缺失的依赖
for package in required_packages:
    try:
        __import__(package if package != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"🔧 正在安装缺失的依赖: {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="High Magnesium Andesite Rock Classification Prediction", layout="wide")

# 1. Load the training dataset
train_file_path = r"D:\JupyterNotebook\lab\FAB-Boninite-HMA-IAT-CA.xlsx"  # Path to the training dataset
train_data = pd.read_excel(train_file_path)

# 2. Data preprocessing
X_train = train_data.drop(train_data.columns[0], axis=1)  # Features
y_train = train_data.iloc[:, 0]  # Labels

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# 3. Build the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_encoded)

# 4. File upload
uploaded_file = st.file_uploader("Upload a new Excel file for prediction", type=["xlsx"])
if uploaded_file is not None:
    input_data = pd.read_excel(uploaded_file)

    # Match column names
    matching_columns = {}
    for col_train in X_train.columns:
        for col_input in input_data.columns:
            if col_input.lower().startswith(col_train.lower()):
                matching_columns[col_train] = col_input
                break
        if col_train not in matching_columns:
            matching_columns[col_train] = None

    # Prepare input data
    X_input = pd.DataFrame()
    for col_train, col_input in matching_columns.items():
        if col_input is not None:
            X_input[col_train] = input_data[col_input]
        else:
            X_input[col_train] = 0  # Fill missing columns with 0

    # 5. Make predictions
    predicted_classes = model.predict(X_input)

    # Get prediction probabilities for each sample
    predicted_probabilities = model.predict_proba(X_input)

    # Get the maximum probability for each sample as confidence
    confidence_scores = np.max(predicted_probabilities, axis=1)

    # Decode the predicted classes
    predicted_classes = label_encoder.inverse_transform(predicted_classes)

    # Add predicted classes and confidence scores to the input data
    input_data['Predicted Class'] = predicted_classes
    input_data['Confidence'] = confidence_scores  # Add confidence column

    # Display the input data with predictions and confidence
    st.write(input_data)

    # 6. Plot the scatter plot with the background image
    # Load background image
    img_path = r"D:\JupyterNotebook\lab\MgO-SiO2.jpg"
    img = Image.open(img_path)

    # Get SiO2 and MgO data
    if 'SiO2' in input_data.columns and 'MgO' in input_data.columns:
        sio2 = input_data['SiO2']
        mgo = input_data['MgO']
    else:
        st.error("The input Excel file is missing SiO2 or MgO columns.")

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[45, 70, 0, 25])

    # Use different colors for different classes
    unique_classes = np.unique(predicted_classes)
    cmap = plt.get_cmap('tab10')
    class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}  # Color map for each class

    for class_name in unique_classes:
        class_indices = predicted_classes == class_name
        plt.scatter(sio2[class_indices], mgo[class_indices], color=class_colors[class_name], label=class_name, alpha=0.6, s=300)

    # Set plot settings
    plt.xlabel('SiO2', fontsize=16)
    plt.ylabel('MgO', fontsize=16)
    plt.title('Scatter Plot of SiO2 and MgO by Class', fontsize=18)
    plt.legend(ncol=5, handletextpad=0.5, columnspacing=1.0, loc='upper right')

    # Hide axes but show labels with larger font
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # Show the plot
    st.pyplot(plt)

    # 7. Generate confidence distribution for each predicted class
    confidences = model.predict_proba(X_input).max(axis=1)  # Get the confidence

    # Plot for each class
    for class_name in unique_classes:
        # Filter confidence data for current class
        class_confidences = confidences[predicted_classes == class_name]

        # Create a new plot for the class
        plt.figure(figsize=(10, 6))

        # Plot the histogram for the current class using the class's color
        plt.hist(class_confidences, bins=20, density=True, alpha=0.7, color=class_colors[class_name], label=f'Predicted Class: {class_name}')

        # Fitting curve (red)
        if len(class_confidences) > 0:
            density = gaussian_kde(class_confidences)
            xs = np.linspace(min(class_confidences), max(class_confidences), 200)
            plt.plot(xs, density(xs), 'r-', label='Fitting Curve')

        plt.xlabel('Confidence', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Confidence Distribution for Class {class_name}', fontsize=16)
        plt.legend()
        plt.grid(True)

        # Save the current class distribution plot
        filename = f'{class_name}_confidence_distribution.png'
        plt.savefig(filename)
        plt.close()

        # Display the plot
        st.image(filename, caption=f'Confidence Distribution for {class_name}')

# Display a message indicating that the model has finished loading and training
st.write("The model has been loaded and trained successfully.")
