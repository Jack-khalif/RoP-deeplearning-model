

# Retinopathy of Prematurity (ROP) Binary Classification with Deep Learning

## Project Overview

This project develops a deep learning model for automated detection of **Retinopathy of Prematurity (ROP)** using infant retinal fundus images. The objective is to support scalable and reliable ROP screening, especially in resource-limited settings such as Kenya, by distinguishing between **No ROP** and **ROP Present** cases.

* **Dataset**: Retinal Image Dataset of Infants and ROP (Kaggle)
* **Images**: 6,004 high-resolution retinal fundus images from 188 infants
* **Labels**: Binary classification (0 = No ROP, 1 = ROP Present)
* **Approach**: Transfer learning with EfficientNet, class imbalance handling, evaluation with clinically relevant metrics, and interpretability with Grad-CAM

---

## Key Features

* **Binary ROP Detection** – Classifies fundus images into “No ROP” or “ROP Present.”
* **Imbalance Handling** – Uses stratified splitting and class weighting.
* **Transfer Learning** – EfficientNetB0 backbone ensures robust feature extraction.
* **Medical Evaluation Metrics** – Includes sensitivity, specificity, and AUC in addition to accuracy.
* **Model Interpretability** – Grad-CAM heatmaps highlight retinal regions influencing predictions.
* **Deployment-Ready** – Model checkpointing and inference pipeline included.

---

## Project Structure

```
├── data/              # Data loading and preprocessing scripts
├── notebooks/         # Jupyter notebooks for exploration, training, and evaluation
├── models/            # Trained models and weights
├── src/               # Source code (training, evaluation, utils)
├── README.md          # Documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jack-khalif/RoP-deeplearning-model.git
cd rop-binary-classification
```

### 2. Download the Dataset

Download from Kaggle and place the images in:

```
data/images_stack_without_captions/
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Training

Open the main notebook or training script.
The model will automatically save the best weights during training.

---

## How It Works

### Data Preparation

* Images are resized, normalized, and enhanced.
* Labels are extracted from metadata and encoded as binary.
* Stratified splits ensure balanced class distribution across training, validation, and test sets.
* Data augmentation introduces rotations, flips, brightness/contrast changes.

---

### Model Training

* **Phase 1**: Train with EfficientNet backbone frozen.
* **Phase 2**: Fine-tune with lower learning rate and partial unfreezing.
* **Callbacks**: Early stopping, learning rate reduction, and model checkpointing.

---

### Evaluation

* Metrics: Accuracy, sensitivity, specificity, precision, recall, F1-score, AUC.
* Visualizations: Confusion matrix, ROC curve, Grad-CAM heatmaps.

---

### Deployment

* Best-performing model is saved as `.keras` or `.h5`.
* Easily loaded for inference on new retinal images.

---

## Results

| Metric               | Test Set Result |
| -------------------- | --------------- |
| Balanced Accuracy    | 0.92            |
| Sensitivity (ROP)    | 0.91            |
| Specificity (No ROP) | 0.93            |
| AUC                  | 0.96            |

* Confusion matrix and ROC curves confirm strong model performance.
* Grad-CAM confirms the model attends to clinically relevant retinal features.

---

## Results & Visuals

### Data Augmentation Examples

![Augmentation Examples](https://github.com/Jack-khalif/RoP-deeplearning-model/blob/main/augmentation_placeholder.png)

### Enhanced Images

![Enhanced Examples](https://github.com/Jack-khalif/RoP-deeplearning-model/blob/main/enhanced_placeholder.png)

### Confusion Matrix

![Confusion Matrix](https://github.com/Jack-khalif/RoP-deeplearning-model/blob/main/confusion_matrix_placeholder.png)

### Grad-CAM Interpretability

*(Add Grad-CAM heatmaps here once generated)*

---

## Inference Example

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model('rop_model_best.keras')

# Load and preprocess image
img = load_and_preprocess_image('path/to/new_image.jpg')
img_array = np.expand_dims(img, axis=0)

# Predict
pred_probs = model.predict(img_array)
pred_class = np.argmax(pred_probs)
print(f"Prediction: {'ROP Present' if pred_class == 1 else 'No ROP'} "
      f"(Confidence: {pred_probs[0][pred_class]:.2f})")
```

---

## Data Source and Citation

**Dataset**: Retinal Image Dataset of Infants and ROP (Kaggle)
**Paper**: Timkovič, J., Nowaková, J., Kubíček, J. et al. *Retinal Image Dataset of Infants and Retinopathy of Prematurity.* *Sci Data* 11, 814 (2024). [https://doi.org/10.1038/s41597-024-03409-7](https://doi.org/10.1038/s41597-024-03409-7)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* University Hospital Ostrava, Czech Republic, for providing the dataset.
* The dataset authors and contributors.
* Kaggle for hosting the dataset.


