Image Classification with TensorFlow (InceptionV3 + Grad-CAM)




A complete deep-learning pipeline for multi-class image classification using InceptionV3 transfer learning, ImageDataGenerator augmentation, and Grad-CAM visualization to interpret predictions.

📌 Features
✔ Transfer Learning using InceptionV3 (ImageNet)
✔ Custom Data Augmentation
✔ Train/Validation Split (70/30)
✔ Training & Validation Plots
✔ Grad-CAM heatmaps for explainability
✔ Classification Report (Precision, Recall, F1-Score)
✔ Fully reproducible pipeline

📂 Dataset Structure

Your dataset must follow this format:
data/
│── class_1/
│── class_2/
│── class_3/
│── ...
Each folder contains images belonging to one class.

🛠️ Tech Stack
Python
TensorFlow / Keras
OpenCV
Matplotlib
NumPy
scikit-learn

📦 Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Dataset Path
Place your dataset in:
/data
Update path in code:
root_dir = 'data/'

🎯 Model Workflow
ImageDataGenerator → InceptionV3 (Frozen) → MaxPooling → Flatten → Dense Softmax

🧠 Transfer Learning Setup
base_model = tf.keras.applications.InceptionV3(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False
✔ Added Custom Layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])
🏋️ Training the Model
model.fit(
    img_generator_flow_train,
    validation_data=img_generator_flow_valid,
    steps_per_epoch=8,
    epochs=40
)
📈 Training Curves
Both accuracy and loss curves are plotted using Matplotlib.
🔥 Grad-CAM Visualization (Explainability)
Grad-CAM highlights the important image regions the model used to make predictions.
Sample Heatmap
The script:
Extracts the final convolution layer (mixed10)
Computes gradients
Generates heatmap
Superimposes it on the original image
Saves output as saved_img.jpg
📊 Evaluation (Classification Report)
from sklearn.metrics import classification_report
print(classification_report(LABEL, PRED))
Outputs:
Precision
Recall
F1-score
Support per class
📁 Recommended Project Structure
├── README.md
├── requirements.txt
├── main.ipynb  /  train.py
├── saved_img.jpg
├── data/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
└── outputs/
    ├── plots/
    ├── gradcam/
🚧 Future Enhancements
Fine-tune last few InceptionV3 layers
Add Dropout for regularization
Use Mixup / CutMix augmentation
Export to TensorFlow Lite for deployment
Build a Streamlit web app for prediction


📜 License
This project is licensed under the MIT License – feel free to use & modify.

👨‍💻 Author
Tanishq Trivedi
Deep Learning | Computer Vision | AI Research

📧 Email: tanusktrivedi@gmail.com
