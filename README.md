# 🏥 Kidney Stone Detection using Vision Transformers

AI-Powered Medical Image Analysis application using Vision Transformers with YOLOv8 for kidney stone detection in CT scans and ultrasound images.

## 🚀 Features

- **Real-time Detection**: Upload medical images and get instant kidney stone detection
- **Confidence Threshold**: Adjustable confidence threshold for detection sensitivity
- **Detailed Results**: View bounding boxes, confidence scores, and class information
- **Export Results**: Download detection results for further analysis
- **Modern UI**: Clean and intuitive Streamlit interface

## 📦 Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vision-Transformers-for-medical-images.git
cd vision-Transformers-for-medical-images
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the app

## 🗂️ Project Structure

```
vision-Transformers-for-medical-images/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── packages.txt          # System packages (for Streamlit Cloud)
├── weights/
│   └── best.pt           # Trained YOLOv8 model weights
├── images/
│   └── STONE- (15).jpg   # Sample images
└── README.md
```

## 🔧 Model Information

- **Architecture**: YOLOv8 with Vision Transformer backbone
- **Training**: Custom trained on kidney stone CT/ultrasound dataset
- **Input**: Medical images (JPG, PNG, BMP, WebP)
- **Output**: Bounding boxes with confidence scores

## 📊 Usage

1. Open the application in your browser
2. Upload a kidney CT scan or ultrasound image using the sidebar
3. Adjust the confidence threshold if needed (default: 0.4)
4. Click "Detect Kidney Stones" to analyze
5. View results and download if needed

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. Always consult healthcare professionals for medical diagnosis. This application should not be used as a substitute for professional medical advice.

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
