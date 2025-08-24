# 🩺 Diabetic Retinopathy Detection  

A deep learning project to detect the severity of **Diabetic Retinopathy** using retinal fundus images.  
The model classifies images into **5 stages**:  
- No DR  
- Mild  
- Moderate  
- Severe  
- Proliferative DR  

Built using **Transfer Learning with ResNet18**.


## 📁 Project Structure

```
Diabetic-Retinopathy-Detection/
│── train.csv
│ 
│
│── train.ipynb
│ 
│
│── models
|      └──retino_model.h5(Generated after training)
│
│── templates/
│ └── index.html
│
│── main.py
│
│
│── .gitignore
│── requirements.txt
│── README.md
```

## 🔍 Dataset
- **Source**: [Kaggle Dataset](https://www.kaggle.com/code/kushalkumar8906kumar/hiee-project/notebook)
- **Directory**: colored_images/ with subfolders for each DR category.
- **Labels**: Provided in train.csv

## 🧠 Model Details
- **Framework**: TensorFlow / Keras
- **Architecture**: ResNet18 via transfer learning
- **Classification Type**: Multiclass (5 classes)
- **Final Model Output**: `retino_model.h5`
- **Achieved Accuracy**: 69%

### How to Run:
### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 2: Train the model
```bash
jupyter notebook notebooks/train.ipynb
```

### Step 3: Run the fastapi webapp
```bash
python src/main.py
```

✅ **Features**

- Classifies 5 stages of diabetic retinopathy
- Uses Transfer Learning with ResNet18
- Flask web interface for predictions
- Based on a real-world medical dataset

