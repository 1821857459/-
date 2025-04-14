import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gaussian_kde
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="High Magnesium Andesite Rock Classification Prediction", layout="wide")

# 1. Load the training dataset
train_file_path = r"data.xlsx"  # <-- 确保此路径正确
train_data = pd.read_excel(train_file_path)

# 2. Data preprocessing
X_train = train_data.drop(train_data.columns[0], axis=1)  # Features
y_train = train_data.iloc[:, 0]  # Labels

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# 3. Build the ensemble model: Random Forest + KNN (k=3)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3)

# Combine using soft voting
model = VotingClassifier(
    estimators=[('rf', rf_model), ('knn', knn_model)],
    voting='soft'
)

# Fit the model
model.fit(X_train, y_train_encoded)

# 4. File upload
uploaded_file = st.file_uploader("Upload a new Excel file for prediction", type=["xlsx"])
if uploaded_file is not None:
    input_data = pd.read_excel(uploaded_file)

    # Match column names (case insensitive, ignore suffixes)
    matching_columns = {}
    processed_train_columns = [col.lower().strip() for col in X_train.columns]
    processed_input_columns = [col.lower().strip() for col in input_data.columns]

    for col_train, processed_col_train in zip(X_train.columns, processed_train_columns):
        matched = False
        for col_input, processed_col_input in zip(input_data.columns, processed_input_columns):
            if processed_col_input.startswith(processed_col_train):
                matching_columns[col_train] = col_input
                matched = True
                break
        if not matched:
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
    predicted_probabilities = model.predict_proba(X_input)
    confidence_scores = np.max(predicted_probabilities, axis=1)
    predicted_classes = label_encoder.inverse_transform(predicted_classes)

    # Append predictions to input data
    input_data['Predicted Class'] = predicted_classes
    input_data['Confidence'] = confidence_scores
    st.write(input_data)

    # 6. Scatter Plot with MgO-SiO2 background image
    img_path = r"MgO-SiO2.jpg"  # <-- 替换为你本地图片路径
    try:
        img = Image.open(img_path)
        if 'SiO2' in input_data.columns and 'MgO' in input_data.columns:
            sio2 = input_data['SiO2']
            mgo = input_data['MgO']
        else:
            st.error("The input Excel file is missing SiO2 or MgO columns.")

        plt.figure(figsize=(10, 10))
        plt.imshow(img, extent=[45, 70, 0, 25])

        unique_classes = np.unique(predicted_classes)
        cmap = plt.get_cmap('tab10')
        class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

        for class_name in unique_classes:
            class_indices = predicted_classes == class_name
            plt.scatter(sio2[class_indices], mgo[class_indices],
                        color=class_colors[class_name], label=class_name, alpha=0.6, s=300)

        plt.xlabel('SiO2', fontsize=16)
        plt.ylabel('MgO', fontsize=16)
        plt.title('Scatter Plot of SiO2 and MgO by Class', fontsize=18)
        plt.legend(ncol=5, loc='upper right')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Failed to load or plot image: {e}")

    # 7. Confidence distribution for each class
    confidences = predicted_probabilities.max(axis=1)
    for class_name in unique_classes:
        class_confidences = confidences[predicted_classes == class_name]
        plt.figure(figsize=(10, 6))
        plt.hist(class_confidences, bins=20, density=True, alpha=0.7,
                 color=class_colors[class_name], label=f'Predicted Class: {class_name}')
        if len(class_confidences) > 0:
            density = gaussian_kde(class_confidences)
            xs = np.linspace(min(class_confidences), max(class_confidences), 200)
            plt.plot(xs, density(xs), 'r-', label='Fitting Curve')
        plt.xlabel('Confidence', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Confidence Distribution for Class {class_name}', fontsize=16)
        plt.legend()
        plt.grid(True)
        filename = f'{class_name}_confidence_distribution.png'
        plt.savefig(filename)
        plt.close()
        st.image(filename, caption=f'Confidence Distribution for {class_name}')

# Final message
st.success("✅ The ensemble model (RandomForest + KNN) has been loaded and trained successfully.")
