# Quick Start Guide

## Financial Report Summarization System

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation Steps

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- Gradio (UI framework)
- PyTorch (deep learning)
- Transformers (HuggingFace models)
- PEFT (parameter-efficient fine-tuning)
- PyPDF2 (PDF processing)
- Other required libraries

#### 2. Verify Model File
Ensure the fine-tuned model exists:
```bash
ls -lh model/fine_tuned_model.pt
```

Expected output: ~1.0 GB file

### Running the Application

#### Method 1: Using the launcher script (Recommended)
```bash
python3 run_app.py
```

#### Method 2: Direct execution
```bash
python3 ui/app.py
```

#### Method 3: From Python
```python
from ui.app import main
main()
```

### Accessing the UI

Once started, the application will display:
```
Financial Report Summarization System
============================================================
Model path: /path/to/model/fine_tuned_model.pt
Starting Gradio interface...
============================================================
Running on local URL:  http://0.0.0.0:7860
```

Open your browser to: **http://localhost:7860**

### Using the Interface

1. **Wait for Model Loading**
   - The model status will show "Model loaded successfully!" when ready
   - This may take 30-60 seconds on first load

2. **Upload PDF**
   - Click "Upload Financial Report (PDF)"
   - Select a financial report PDF file
   - Supported: Any PDF with extractable text

3. **Generate Summary**
   - Click "Generate Summary" button
   - Wait for processing (10-30 seconds depending on document size)

4. **View Results**
   - **Extracted Text Preview**: Shows first 500 characters of extracted text
   - **Generated Summary**: AI-generated summary of the financial report

### Google Colab Usage

If running in Google Colab:

```python
# Install dependencies
!pip install -r requirements.txt

# Run the app with public sharing
!python run_app.py
```

Then modify `ui/app.py` line 159 to set `share=True`:
```python
app.launch(share=True, ...)
```

This creates a public URL you can access from anywhere.

### Troubleshooting

**Problem: "Model file not found"**
- Solution: Verify `model/fine_tuned_model.pt` exists and is ~1GB

**Problem: "Out of memory"**
- Solution: Close other applications, or use CPU mode by modifying inference.py

**Problem: "PDF extraction failed"**
- Solution: Ensure PDF is not encrypted and contains text (not scanned images)

**Problem: Port 7860 already in use**
- Solution: Change port in `ui/app.py` line 160 to a different number (e.g., 7861)

### System Requirements

**Minimum:**
- 4GB RAM
- CPU: Any modern processor
- Storage: 5GB free space

**Recommended:**
- 8GB+ RAM
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: 10GB free space

### Performance Notes

- **First run**: Downloads base TinyLlama model (~2.2GB) from HuggingFace
- **CPU mode**: 20-60 seconds per summary
- **GPU mode**: 5-15 seconds per summary
- **Document size**: Longer documents take more time

### Next Steps

- Test with sample financial reports
- Adjust `max_new_tokens` in `utils/inference.py` for longer/shorter summaries
- Modify the system prompt for different summarization styles
- Add custom preprocessing for specific document formats

### Support

For issues or questions:
1. Check the main README.md
2. Review error messages in the terminal
3. Verify all dependencies are installed correctly
