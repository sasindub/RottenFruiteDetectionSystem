# Advanced Fruit Quality Detection System

This project is an **Advanced Fruit Quality Detection System** that uses **deep learning and computer vision** to classify fruits as **fresh** or **rotten**. The system currently supports **apples, bananas, and oranges**. It demonstrates the full pipeline of preprocessing, model training, evaluation, and deployment.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Technologies Used](#technologies-used)  
- [Setup Instructions](#setup-instructions)  
- [Usage](#usage)  
- [Model Details](#model-details)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Challenges](#challenges)  
- [Future Improvements](#future-improvements)

---

## Project Overview

Fruits are perishable and their quality directly affects human health and economy. This system automates **fresh vs rotten classification** using image data. The goal is to help farmers, sellers, and consumers quickly assess fruit quality without manual inspection.

The system involves:

1. **Preprocessing images** – resizing, normalization, and data augmentation.  
2. **Training deep learning models** – Simple CNN and MobileNetV2.  
3. **Evaluating performance** – using accuracy, precision, recall, and F1-score.  
4. **Deploying the best model** – for real-time classification of new fruit images.

---

## Dataset

The dataset is organized into:

dataset/
├── Train/
│ ├── freshapples/
│ ├── rottenapples/
│ ├── freshbanana/
│ ├── rottenbanana/
│ ├── freshoranges/
│ ├── rottenoranges/
├── Test/
│ ├── freshapples/
│ ├── rottenapples/
│ ├── freshbanana/
│ ├── rottenbanana/
│ ├── freshoranges/
│ ├── rottenoranges/



> Note: Dataset was reduced for training on a low-spec machine to speed up training.

---

## Technologies Used

- **Python 3.10+**  
- **TensorFlow / Keras** – for deep learning model building  
- **NumPy, PIL** – for image processing  
- **Scikit-learn** – for evaluation metrics  
- **FastAPI** – for deploying the model as a web service  
- **Bootstrap** – for front-end user interface  

---

## Setup Instructions

1. **Clone the repository**


git clone https://github.com/<your-username>/AdvancedFruitQualityDetection.git
cd AdvancedFruitQualityDetection
Create a virtual environment


python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Install dependencies


pip install -r requirements.txt
Ensure dataset folder is in project root
The folder structure should match the Dataset section above.

Train the models (Optional if you want to retrain)


python model.py
This will train both Simple CNN and MobileNetV2 and save the best model as best_model.h5.

Run the FastAPI application


uvicorn app:app --reload
Access the web interface
Open your browser at http://127.0.0.1:8000

Usage
Upload an image of a fruit (apple, banana, or orange).

The system will predict whether the fruit is fresh or rotten.

The result will display the predicted class and confidence score.

Model Details
Model 1: Simple CNN
Convolutional layers with ReLU activations

MaxPooling layers for downsampling

Dropout layer for regularization

Dense layers with softmax for classification

Model 2: MobileNetV2 (Transfer Learning)
Pretrained on ImageNet

Feature extraction frozen initially

Global Average Pooling + Dense layer for classification

Dropout for regularization

Best Model: Simple CNN (achieved ~98% accuracy on the test set)

Evaluation Metrics
Accuracy: 0.98

Precision, Recall, F1-Score: Calculated per fruit class

Confusion matrix and classification report generated for detailed performance

Place screenshots of metrics and confusion matrices here

Challenges
Limited machine specs forced dataset reduction

MobileNetV2 performed worse due to small dataset size

Choosing the best model required multiple experiments

Long training time for high-resolution images

Future Improvements
Train on larger dataset for better generalization

Add more fruit categories

Integrate real-time camera feed for live classification

Deploy on cloud for mobile or web access

License
This project is for academic purposes. No commercial use allowed.

Author
Sasindu Bandara – BSc Computing Student, Sri Lanka
