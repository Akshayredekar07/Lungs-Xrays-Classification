# Pneumonia Detection with Gradio

A Gradio-based application for predicting pneumonia from chest X-ray images using a pre-trained EfficientNet-B0 model.

## Setup (Windows, Manual)

1. **Clone the repository** (if using Git):
   ```cmd
   git clone <repository-url>
   cd Lungs-Xrays-Classification
   ```
   Or download and extract the project folder to `D:\Pytorch\Lungs-Xrays-Classification`.

2. **Place the model**:
   - Copy `pneumonia_model.pth` to `models`.

3. **Set up the environment**:
   - Open Command Prompt and navigate to the project directory:
     ```cmd
     cd D:\Pytorch\Lungs-Xrays-Classification
     ```
   - Create a virtual environment:
     ```cmd
     python -m venv .venv
     ```
   - Activate the virtual environment:
     ```cmd
     .venv\Scripts\activate
     ```
   - Install dependencies:
     ```cmd
     pip install -r requirements.txt
     ```

4. **Run the application**:
   - Ensure the virtual environment is activated:
     ```cmd
     .venv\Scripts\activate
     ```
   - Run the Gradio app:
     ```cmd
     python app.py
     ```
   - The interface will be available at `http://localhost:7860`.
