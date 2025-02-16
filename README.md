# Career Predictor - Streamlit Web App

## Overview
This project is a **Career Predictor** web app built using **Streamlit** and a trained **Neural Network model**. It allows users to select their skills from a dropdown list, and based on their choices, it predicts the most suitable career.

## Features
- **Skill-based Career Prediction**: Uses a trained **TensorFlow/Keras model**.
- **Interactive UI**: Select skills dynamically.
- **Relaxing UI Design**: A calming blue background with a bubble effect.
- **Instant Prediction**: Provides real-time predictions for careers.

## Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/career-predictor.git
cd career-predictor
```

### **2. Install Dependencies**
Make sure you have Python 3.8+ installed. Then, run:
```bash
pip install -r requirements.txt
```

### **3. Run the App**
```bash
streamlit run app.py
```

## Model Training (Optional)
If you need to retrain the model, follow these steps:
```python
python train_model.py
```
This will generate a new `career_nn_model.h5` and `label_encoder.pkl`.

## Deployment
You can deploy the app on **Streamlit Cloud** or **AWS/GCP**.

## File Structure
```
career-predictor/
│── app.py                 # Main Streamlit application
│── career_nn_model.h5     # Trained neural network model
│── label_encoder.pkl      # Label encoder for career mapping
│── requirements.txt       # Dependencies
│── train_model.py         # Model training script (optional)
│── processed_career_dataset.csv # Preprocessed dataset
│── README.md              # This readme file
```

## Contributing
Feel free to fork the repository and submit pull requests! 🚀

## License
MIT License

---

### Need Help?
Reach out to `your-email@example.com` for any questions!
