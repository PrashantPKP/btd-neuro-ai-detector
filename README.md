# 🧠 NeuroAI Detector - Brain Tumor Detection System

**AI-Powered Brain Tumor Detection using Deep Learning and MRI Analysis**

A cutting-edge web application that leverages artificial intelligence to detect brain tumors from MRI scans with **97% accuracy**. Built with Flask backend and interactive web interface, this system assists medical professionals in early diagnosis and treatment planning.

---

## ✨ Features

### 🚀 Core Features
- **AI-Powered Detection**: Deep Learning CNN model trained on thousands of MRI scans
- **High Accuracy**: 97% accuracy rate on validated datasets
- **Instant Results**: Get analysis results in less than 5 seconds
- **User-Friendly Interface**: Intuitive web interface for easy image upload and analysis
- **Medical-Grade Analysis**: Processes 8 million parameters for comprehensive evaluation
- **Patient Information Tracking**: Store and display patient name and analysis details
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

### 🎨 UI/UX Features
- **Modern Bootstrap 5 Design**: Clean, professional interface
- **Animated Components**: Smooth transitions and floating shapes for visual appeal
- **Real-time Results Display**: Comprehensive result visualization with confidence metrics
- **Downloadable Reports**: Print-friendly result summaries for medical records
- **Navigation Menu**: Quick access to home, features, analysis, and resources
- **Social Media Integration**: Links to developer's professional profiles

### 🔒 Security & Privacy
- **Secure Processing**: Images processed securely without permanent storage
- **Data Privacy**: Uploaded images are temporary and cleared after analysis
- **Medical Compliance**: Designed with healthcare privacy in mind

---

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB for model and dependencies
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster processing

### Software
- Windows 10/11, macOS, or Linux
- pip (Python Package Manager)
- Virtual Environment (recommended)

---

## 📦 Installation

### Step 1: Clone or Download the Repository

```bash
# Clone the repository
git clone https://github.com/PrashantPKP/btd-neuro-ai-detector.git
cd "btd-neuro-ai-detector"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow; import flask; import cv2; print('All dependencies installed successfully!')"
```

---

## 🗂️ Dataset Setup

### Download Dataset from Kaggle

The application uses the **Brain Tumor Detection dataset** from Kaggle.

**Dataset Link**: [Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

#### Steps to Download:

1. **Create Kaggle Account**
   - Visit [kaggle.com](https://www.kaggle.com)
   - Sign up for a free account

2. **Download Dataset**
   - Go to the dataset link above
   - Click "Download" button
   - Extract the downloaded ZIP file

3. **Dataset Information**
   - **Total Images**: 7,000+ MRI scans
   - **Image Format**: JPG, PNG
   - **Image Size**: Typically 224x224 pixels
   - **Classes**: No Tumor and Tumor
   - **Split**: Training/Testing ratio

---

## 🚀 Running the Application

### Quick Start

```bash
# Make sure virtual environment is activated
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Run the Flask application
python app.py
```

### Access the Web Interface

1. Open your web browser
2. Navigate to: `http://localhost:5000`
3. You should see the NeuroAI Detector homepage

### Running on Different Host/Port

```bash
# Run on specific port
python app.py --port 8000

# Run on network (accessible from other machines)
# Edit app.py and change: app.run(host='0.0.0.0', port=5000)
```


## 🖱️ How to Use

### Step 1: Access the Application
- Open `http://localhost:5000` in your web browser
- You'll see the NeuroAI Detector homepage

### Step 2: Navigate to Analysis
- Click on "**Analyze**" in the navigation menu or "**Start Analysis**" button
- Scroll to the upload section

### Step 3: Upload MRI Scan
1. Enter the **Patient Name** (or test identifier)
2. Select an MRI scan image file:
   - Accepted formats: JPG, PNG
   - Recommended size: 224x224 pixels
   - Maximum file size: 10MB
3. Click "**Analyze Image**"

### Step 4: View Results
- The system will process the image
- Results display includes:
  - **Tumor Status**: "TUMOR DETECTED" or "NO TUMOR DETECTED"
  - **AI Confidence**: Confidence level of prediction
  - **Technical Metrics**: Model accuracy, parameters, processing time
  - **Patient Information**: Name and analysis timestamp
  - **Analyzed Image**: Display of processed MRI scan

### Step 5: Further Actions
- **New Analysis**: Perform another analysis
- **Print Results**: Print the results for medical records
- **Medical Resources**: Access additional medical information

---

## 🔧 Technical Details

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask 3.1.2 |
| **Deep Learning** | TensorFlow 2.10.0, Keras 2.10.0 |
| **Image Processing** | OpenCV 4.12.0, Pillow 11.3.0 |
| **Frontend** | HTML5, CSS3, Bootstrap 5, JavaScript |
| **Web Server** | Werkzeug 3.1.3 |
---


## ⚕️ Disclaimer

**IMPORTANT MEDICAL NOTICE:**

This AI analysis tool is designed to **assist medical professionals only** and should **NOT** be used as:
- A replacement for professional medical diagnosis
- Standalone clinical decision support
- Medical treatment recommendation tool
- Primary diagnostic tool

**Always consult with qualified healthcare providers** for:
- Proper diagnosis and interpretation
- Treatment planning and decisions
- Medical management of brain tumors

**Early detection and professional medical care are crucial** for optimal patient outcomes. This tool is intended to supplement, not replace, professional medical judgment.

---

## 📚 Resources

### Medical Information
- [Brain Tumor Diagnosis - Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/brain-tumor/diagnosis-treatment/drc-20350088)
- [Understanding Brain Tumors - Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084)
- [MRI Technology for Brain Imaging](https://kidshealth.org/en/parents/mri-brain.html)
- [Brain Tumor Treatment Options](https://www.cancer.net/cancer-types/brain-tumor/diagnosis)

### Technical Resources
- [CNN in Medical Imaging](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Models](https://keras.io/models/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Dataset
- [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

### Source Code
- [GitHub Repository](https://github.com/PrashantPKP/btd-neuro-ai-detector)

---

## 👨‍💻 Author

**Prashant Parshuramkar**

- 🔗 [GitHub](https://github.com/PrashantPKP)
- 💼 [LinkedIn](https://www.linkedin.com/in/prashantpkp/)
- 🌐 [Portfolio](https://prashantparshuramkar.host20.uk/)

Built with ❤️ for medical innovation and early disease detection

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Fork the repository


---

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section above
2. Review the GitHub repository issues
3. Contact the developer through GitHub




