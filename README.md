# Plant Disease Detection using Deep Learning

A web-based plant disease classification system built with **Convolutional Neural Networks (CNN)** and deployed using **Streamlit**.  
The model identifies 38 different classes of plant diseases (and healthy states) across 14 crop species.

![Plant Disease Demo](Disease.png)

## Features

- Supports 38 plant disease / healthy classes (Apple, Tomato, Potato, Corn, Grape, etc.)
- Image upload and real-time prediction via Streamlit web interface
- Data augmentation during training (rotation, flip, zoom, etc.)
- Trained on the **New Plant Diseases Dataset (Augmented)**
- Clean preprocessing pipeline with OpenCV + TensorFlow/Keras
- Training history visualization (loss & accuracy curves)

## Dataset

The model was trained on the **New Plant Diseases Dataset (Augmented)** containing ≈ 87,900 images across 38 classes.

**Download the dataset here:**  
[🔗 Dataset Download Link](https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset?dataset_version_number=2)  <!-- ← Replace with actual link -->

Original source:  
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

## Pretrained Model

**Download the trained model (.keras format):**  
[🔗 Download Pretrained Model](https://github.com/shrey2199/plant_disease_predictor/releases/download/v1.0/plant_disease_cnn_model.keras) (Also available in Github Releases)  <!-- ← Replace with actual link (GitHub release, Drive, HF, etc.) -->

## Technologies Used

- Python 3.9+
- TensorFlow / Keras
- Streamlit
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow (PIL)
- kagglehub (for dataset download in notebook)

## Project Structure

```
plant-disease-detection/
├── app.py                        # Streamlit web application
├── PlantDiseaseDetection.ipynb   # Jupyter notebook used for training
├── plant_disease_cnn_model.keras # Trained model file (not in git – download separately)
├── Disease.png                   # Demo image used in app & README
├── requirements.txt
├── README.md
└── (new_plant_diseases/          # dataset folder – not included in repo)
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/plant-disease-detection.git
cd plant-disease-detection
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# or
venv\Scripts\activate         # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage – Run the App

```bash
streamlit run app.py
```

Open your browser at:  
http://localhost:8501

1. Upload a plant leaf image
2. Click **Show Image** (optional)
3. Click **Predict**

## Training (Reproducing the model – optional)

1. Open `PlantDiseaseDetection.ipynb` in Jupyter, VSCode, or Colab
2. Download the dataset using the link above (or kagglehub in the notebook)
3. Adjust paths to `train` and `valid` folders if needed
4. Run all cells → model will be saved as `plant_disease_cnn_model.keras`

## Results Summary (from your notebook logs)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc  |
|-------|------------|----------|-----------|----------|
| 1     | 2.7125     | 1.2689   | 24.85%    | 59.64%   |
| 2     | 0.8645     | 0.6178   | 72.88%    | 79.85%   |
| 3     | 0.4605     | 0.4053   | 84.93%    | 87.43%   |
| 4     | 0.3253     | 0.3354   | 89.35%    | 89.67%   |
| 5     | 0.2569     | 0.2926   | 91.48%    | 91.04%   |

→ Final validation accuracy: **~91%**

## Acknowledgments

- Dataset: [New Plant Diseases Dataset – Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)