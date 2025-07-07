# 📱 Mobile Model Analysis (Flipkart India)  
A Data Science Project analyzing smartphone models from Flipkart’s Indian marketplace using an interactive **Streamlit** app.  

This project aims to uncover brand-wise, feature-wise, and model-wise insights through comprehensive preprocessing, EDA, and visualizations.

The app is live at: (https://mobile-model-analysis.streamlit.app/)

---

## 🔍 Project Goals

- Clean and preprocess raw mobile phone data from Flipkart.
- Perform detailed **EDA** (Exploratory Data Analysis) covering:
  - Brand-wise trends
  - Model-wise comparisons
  - Feature-price relationships
  - Market segmentation (Budget / Mid-range / Flagship)
- Build an interactive and user-friendly **Streamlit** dashboard with:
  - Data preprocessing pipeline
  - EDA visualizations
  - Brand Insights
  - Model comparison interface

---

## 📁 Dataset

- Source: [Flipkart (Kaggle Dataset)](https://www.kaggle.com/datasets/mrmars1010/filpkart-mobiles)
- Focus: Indian smartphone market
- Format: CSV
- Columns include:
  - Product Name, Actual Price, Discount Price, Ratings, Reviews
  - RAM, Storage, Display Size, Camera Specs, etc.

---

## ⚙️ Preprocessing Steps

1. Extract brand names from product names (with exceptions like `I kall` → `IKall`).
2. Clean and convert price, rating, and review columns.
3. Parse and extract numerical features like RAM, Storage, Camera MP.
4. Fill missing values using **brand-segment-wise medians or modes**.
5. Classify models into `Budget`, `Mid-Range`, or `Flagship` segments.
6. Save cleaned dataset to `Cleaned_Mobiles_Dataset.csv`.

---

## 📊 Streamlit App Features

### 1. **Overview Tab**
- Raw dataset preview  
- Project introduction  

### 2. **Preprocessing Tab**
- Run preprocessing on raw data  
- View cleaned and transformed dataset  

### 4. **Brand Insights Tab**
- Select a brand to explore:
  - Segment distribution
  - Feature vs Price (RAM, Storage, Camera)
  - Brand-specific statistics

### 5. **Compare Models**
- Select multiple models (even across brands)  
- Compare features like:
  - Price, RAM, Storage
  - Ratings & Reviews (Bubble Chart)
- Add/Remove selections interactively

### 6. **Feature Insight**
- Analyse what dependency the whole dataset shows between different attributes

---

## 📦 How to Run the App

1. Clone the repository  
2. Install dependencies:

    pip install -r requirements.txt
3.  Launch the Streamlit app:

    streamlit run app.py

---
## 📌 Future Enhancements
Add prediction or clustering for price segment classification

Integrate web scraping or periodic dataset update

Improve model performance insights using ML algorithms

---

## 🤝 Credits
This project was developed by Aman Singh as part of an internship at O7 Services.
It was built under the valuable guidance of Mr. Harkirat Singh and Mr. Haritesh Chauhan, whose mentorship played a key role in shaping the direction and quality of the work.

Special thanks to O7 Services for providing the opportunity and resources to undertake this project.