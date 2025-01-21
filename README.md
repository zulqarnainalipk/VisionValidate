# **VisionValidate: Real vs Fake Autonomous Driving Scene Classifier**  
A deep learning-based project to classify autonomous driving scenes as **real** or **fake** using state-of-the-art computer vision techniques.

---


![Project Banner](vision%20validate.jpeg)





## **Overview**  
**VisionValidate** is a project aimed at developing a robust binary classification model to distinguish between real and fake images of autonomous driving scenarios. By leveraging modern transformer-based architectures, advanced augmentations, and efficient training techniques, the model predicts the probability of an image being real (1) or fake (0).

The core objective of this project is to demonstrate the application of deep learning in solving challenges related to autonomous driving image classification.

---

## **Dataset**  
- **Input**: RGB images in JPEG format.  
- **Labels**: Binary classification (`1` for real, `0` for fake).  
- **Structure**:  
  - `train.csv`: Contains file paths and corresponding labels for training data.  
  - `Train/`: Folder containing training images.  
  - `Test/`: Folder containing test images (labels not provided).  
  - `sample_submission.csv`: Template file for test predictions.  

---

## **Approach**  

### **1. Model**  
The core of **VisionValidate** is the **Swin Transformer**, a state-of-the-art vision transformer model:  
- Pretrained on ImageNet for transfer learning.  
- Fine-tuned for the binary classification task.  

### **2. Data Augmentation**  
To improve robustness and generalization, **Albumentations** was employed for augmentations:  
- **Training Augmentations**:  
  - Resizing to 224x224 pixels.  
  - Random horizontal flips.  
  - Brightness and contrast adjustments.  
  - Gaussian blur.  
  - Hue and saturation modifications.  
  - Normalization using ImageNet statistics.  
- **Validation Augmentations**:  
  - Resizing and normalization only.

### **3. Cross-Validation**  
Implemented **Stratified K-Fold Cross-Validation** to ensure balanced class distribution and robust evaluation.  

### **4. Training Pipeline**  
- **Criterion**: Binary Cross-Entropy with Logits Loss.  
- **Optimizer**: AdamW for efficient optimization of large-scale transformer models.  
- **Scheduler**: OneCycleLR for dynamic learning rate adjustment.  
- **Mixed Precision**: Enabled PyTorch AMP (Automatic Mixed Precision) for faster training and reduced memory usage.  

### **5. Metrics**  
The primary metric for evaluation is **AUC-ROC**. Additionally:  
- **Accuracy**: To evaluate the overall correctness of predictions.  
- **F1-Score**: To measure the balance between precision and recall.  

---

## **Use Cases and Applications**  

**VisionValidate** can be applied in various domains, particularly those focused on improving the reliability and safety of autonomous driving systems. Here are some of the key potential use cases:

### **1. Autonomous Vehicle Systems**  
- **Scene Verification**: **VisionValidate** can be used to verify the authenticity of driving scenes captured by sensors in autonomous vehicles. By ensuring that the driving scenes are real, autonomous vehicles can avoid simulated or manipulated scenarios that may lead to incorrect decision-making.
  
### **2. AI Safety and Validation**  
- **AI Training Data Quality Control**: Ensuring that training data used to train AI models for autonomous driving contains real-world, accurate scenes rather than fake or manipulated data. **VisionValidate** can help identify and filter out fake data, leading to better-trained AI models for safer driving systems.
  
### **3. Simulation vs. Real-World Data**  
- **Improved Simulation Data**: For autonomous vehicle manufacturers, it's critical to differentiate between real-world and simulated driving data. By using **VisionValidate**, manufacturers can assess the quality of their training datasets and filter out any images that may not represent actual driving conditions, ensuring their models learn to handle real-world scenarios.
  
### **4. Image Data Analysis for Research and Development**  
- **Data Cleaning in Research**: For researchers working with datasets involving autonomous driving, **VisionValidate** can be used to classify and clean datasets, ensuring only valid, real images are used for experimentation.
  
### **5. Autonomous Driving Datasets**  
- **Data Labeling and Quality Assurance**: For large-scale autonomous driving datasets, **VisionValidate** can help in automatically labeling or flagging fake or poor-quality images. It ensures that the data used for training autonomous driving models is of the highest quality.

---

## **How VisionValidate Helps**  

1. **Improves Training Data Quality**:  
   By distinguishing between real and fake driving scenarios, **VisionValidate** can help improve the quality of datasets used for training autonomous systems, ultimately making models more robust and reliable.

2. **Boosts Safety in Autonomous Systems**:  
   Ensuring that the data used in autonomous driving systems is accurate and representative of real-world scenarios can prevent safety hazards caused by incorrect training data. This project is a step toward creating safer AI systems for autonomous vehicles.

3. **Facilitates Trustworthy AI Systems**:  
   As autonomous vehicles and AI systems become more prevalent, their trustworthiness is of utmost importance. **VisionValidate** helps in verifying the authenticity of driving scenes, ensuring that the AI models make decisions based on real, trustworthy data.

4. **Automates Data Validation**:  
   **VisionValidate** automates the process of classifying driving scenes, providing an efficient tool to evaluate and validate large amounts of image data quickly. This is useful for both industry applications and research purposes.

---

## **Code Overview**  

### **1. Dataset Class**  
The `CustomImageDataset` class handles image loading and transformation using PyTorch and Albumentations.

### **2. Transformations**  
Defined separate transformation pipelines for training and validation data using `A.Compose`.  

### **3. Data Loaders**  
Created train and validation loaders using PyTorch’s `DataLoader` and `Subset` functionalities.

### **4. Training Loop**  
The training loop:  
- Computes forward passes using mixed precision.  
- Optimizes weights using AdamW and updates learning rates with OneCycleLR.  
- Logs training loss at each epoch.  

### **5. Validation Loop**  
The validation loop:  
- Evaluates the model on validation data.  
- Computes loss, predictions, and metrics such as AUC-ROC and F1-Score.  

### **6. Inference**  
- Loads trained weights for inference on the test dataset.  
- Applies transformations to test images and generates predictions.  

---

## **Usage Instructions**  

### **1. Install Dependencies**  
```bash
pip install torch torchvision timm albumentations scikit-learn pandas pillow matplotlib tqdm
```

### **2. Prepare Data**  
- Place your training images in a folder named `Train/`.  
- Ensure your test images are in a folder named `Test/`.  
- Provide a CSV file (`train.csv`) with the following structure:  
  ```
  filename,label
  image1.jpg,1
  image2.jpg,0
  ```

### **3. Train the Model**  
Run the training script:  
```bash
python main.py
```
- Modify parameters like `img_size`, `batch_size`, `num_epochs`, and `lr` in the `train_model` function as needed.  
- The best model will be saved as `best_model_fold{fold_num}.pth`.  

### **4. Predict on Test Data**  
Generate predictions for test images:  
```bash
python main.py
```
Predictions will be saved to `submission.csv`.  

---

## **Results**  
- **Model Architecture**: Swin Transformer.  
- **Metrics**:  
  - AUC-ROC: Achieved high accuracy in distinguishing between real and fake images.  
  - Robust performance across multiple validation folds.  

---

## **Lessons Learned**  
1. **Transformer Models**: Leveraging pretrained transformer models like Swin Transformer can significantly improve results in image classification tasks.  
2. **Augmentations**: Data augmentation plays a crucial role in enhancing model generalization.  
3. **Metric Alignment**: Optimizing directly for AUC-ROC aligns the training process with project goals.  
4. **Cross-Validation**: Stratified K-Fold validation ensures consistent evaluation and prevents overfitting.  

---

## **Future Work**  
- Experiment with ensemble models for improved predictions.  
- Explore other advanced architectures like ConvNeXt or EfficientNet.  
- Apply self-supervised learning to utilize unlabeled data more effectively.  

---

## **License**  
This project is licensed under the MIT License.  

---

## Want to ask anything? Connect with me on [LinkedIn](https://www.linkedin.com/in/zulqarnainalipk/).
