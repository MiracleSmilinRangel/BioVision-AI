

\# 🧬 BioVision AI

\### AI-Powered Contactless Health Monitoring System



BioVision AI is an experimental computer vision project that uses a webcam to estimate selected health-related signals without physical contact.



\## 🚀 Features



\### 1. Blink \& Fatigue Dynamics

\- Detects eye activity using facial landmarks.

\- Uses Eye Aspect Ratio (EAR).

\- Tracks blink patterns and blink rate.



\### 2. Respiration Rate Estimation

\- Uses upper-body visual signals.

\- Estimates breathing rate through image processing.



\### 3. Heart Rate Estimation (rPPG)

\- Extracts facial colour variations.

\- Uses remote photoplethysmography (rPPG) techniques to estimate heart rate.



\## 🛠️ Technologies Used



\- Python

\- OpenCV

\- MediaPipe

\- NumPy

\- Pandas

\- SciPy

\- Streamlit

\- Streamlit-WebRTC



\## 📁 Project Structure



```text

BioVision-AI/

├── models/

│   ├── face\_landmarker.task

│   └── pose\_landmarker\_lite.task

├── biovision\_live.py

├── requirements.txt

├── .gitignore

└── README.md

```



\## ⚙️ Installation and Setup



1\. Clone the repository:



```bash

git clone https://github.com/MiracleSmilinRangel/BioVision-AI.git

cd BioVision-AI

```



2\. Create and activate a virtual environment:



```bash

python -m venv venv

venv\\Scripts\\activate

```



3\. Install dependencies:



```bash

pip install -r requirements.txt

```



4\. Ensure both required model files are available inside the `models/` folder.



5\. Run the application:



```bash

streamlit run biovision\_live.py

```



\## ⚠️ Disclaimer



This project is developed for educational and experimental purposes. The outputs are algorithmic estimates and are not medically validated. They should not be used for medical diagnosis or treatment.



\## 👩‍💻 Developed With



Python | Computer Vision | Artificial Intelligence



