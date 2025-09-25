Retinopathy of Prematurity (ROP) Binary Classification with Deep Learning
Project Overview
This project builds a deep learning model to automatically detect Retinopathy of Prematurity (ROP) from retinal fundus images of infants. The goal is to enable scalable, accurate ROP screening—especially in resource-limited settings like Kenya—by distinguishing between "No ROP" and "ROP Present" cases.

Dataset: Retinal Image Dataset of Infants and ROP (Kaggle)

Images: 6,004 high-resolution retinal images from 188 infants

Labels: Binary (0 = No ROP, 1 = ROP Present)

Approach: Transfer learning with EfficientNet, class imbalance handling, medical evaluation metrics, and model interpretability

 Features
Binary ROP detection: Classifies images as "No ROP" or "ROP Present"

Handles severe class imbalance: Uses class weighting and stratified splits

Transfer learning: EfficientNetB0 backbone for robust feature extraction

Medical metrics: Evaluates sensitivity, specificity, AUC, and more

Model interpretability: Grad-CAM visualizations to show model focus

Ready for deployment: Model saving and inference pipeline included

 Project Structure
text
├── data/                         # Data loading and preprocessing scripts
├── notebooks/                    # Jupyter notebooks for EDA, training, evaluation
├── models/                       # Saved models and weights
├── src/                          # Source code for training, evaluation, utils
├── README.md                     # Project documentation (this file)
 Getting Started
1. Clone the repository
bash
git clone https://github.com/yourusername/rop-binary-classification.git
cd rop-binary-classification
2. Download the Dataset
Add the Kaggle dataset to your environment or download it from here.

Place the images in data/images_stack_without_captions/.

3. Install Dependencies
bash
pip install -r requirements.txt
4. Run Training
Open the main notebook or script and follow the step-by-step instructions.

The model will automatically save the best weights during training.

 How It Works
Data Preparation:

Images are loaded and preprocessed (resized, enhanced, normalized).

Labels are extracted from filenames or metadata and converted to binary.

Stratified train/val/test splits ensure balanced evaluation.

Model Training:

Phase 1: Train EfficientNet with frozen base layers and medical class weights.

Phase 2: Fine-tune by unfreezing base layers and lowering the learning rate.

Callbacks handle early stopping, checkpointing, and learning rate scheduling.

Evaluation:

Reports accuracy, precision, recall (sensitivity), specificity, and AUC.

Plots confusion matrix and ROC curve.

Interpretability:

Grad-CAM heatmaps show which retinal regions influenced the model's decision.

Deployment:

The best model is saved and can be loaded for inference on new images.

 Results
Balanced accuracy on test set (example): 0.92

Sensitivity (Recall for ROP): 0.91

Specificity (Recall for No ROP): 0.93

AUC: 0.96

Interpretability: Grad-CAM visualizations confirm model focuses on relevant retinal features

Inference Example
python
from tensorflow.keras.models import load_model
img = load_and_preprocess_image('path/to/new_image.jpg')
img_array = np.expand_dims(img, axis=0)
model = load_model('best_rop_model.h5')
pred_probs = model.predict(img_array)
pred_class = np.argmax(pred_probs)
print(f"Prediction: {'ROP Present' if pred_class == 1 else 'No ROP'} (Confidence: {pred_probs[0][pred_class]:.2f})")
 Data Source & Citation
Dataset: Retinal Image Dataset of Infants and ROP (Kaggle)

Paper: Timkovič, J., Nowaková, J., Kubíček, J. et al. Retinal Image Dataset of Infants and Retinopathy of Prematurity. Sci Data 11, 814 (2024). https://doi.org/10.1038/s41597-024-03409-7

 Contributing
Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

 Acknowledgements
The University Hospital Ostrava, Czech Republic, for the dataset

The original dataset authors and contributors

Kaggle for hosting the data
