# System Architecture

## Financial Report Summarization System

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                      (Gradio Web UI)                         │
│                         ui/app.py                            │
└────────────────┬────────────────────────────┬────────────────┘
                 │                            │
                 ▼                            ▼
    ┌────────────────────────┐   ┌────────────────────────┐
    │   PDF Text Extraction  │   │  Model Inference       │
    │   utils/pdf_loader.py  │   │  utils/inference.py    │
    └────────────────────────┘   └───────────┬────────────┘
                 │                            │
                 │                            ▼
                 │               ┌────────────────────────┐
                 │               │  Fine-tuned Model      │
                 │               │  model/                │
                 │               │  fine_tuned_model.pt   │
                 │               └────────────────────────┘
                 │
                 ▼
         [PDF Document]
```

### Component Details

#### 1. User Interface Layer (`ui/app.py`)
**Purpose**: Provide web-based interface for user interaction

**Key Functions**:
- `create_ui()`: Build Gradio interface components
- `process_pdf_and_summarize()`: Main processing pipeline
- `initialize_model()`: Load model on startup

**Components**:
- File upload widget (PDF input)
- Submit button (trigger processing)
- Text preview output (extracted text)
- Summary output (AI-generated summary)
- Status indicator (model loading state)

**Flow**:
1. User uploads PDF file
2. User clicks "Generate Summary"
3. System extracts text from PDF
4. System generates summary using model
5. Results displayed to user

---

#### 2. PDF Processing Layer (`utils/pdf_loader.py`)
**Purpose**: Extract text content from PDF files

**Key Functions**:
- `extract_text_from_pdf(pdf_path)`: Extract all text from PDF
- `validate_pdf(pdf_path)`: Check if file is valid PDF

**Process**:
1. Open PDF file using PyPDF2
2. Iterate through all pages
3. Extract text from each page
4. Combine and clean text
5. Return processed text

**Error Handling**:
- Invalid PDF format
- Encrypted/protected PDFs
- Empty or unreadable PDFs

---

#### 3. Model Inference Layer (`utils/inference.py`)
**Purpose**: Load model and generate summaries

**Key Classes**:
- `FinancialSummarizer`: Main model wrapper class

**Key Functions**:
- `load_model()`: Initialize model and tokenizer
- `format_prompt()`: Create instruction-style prompt
- `generate_summary()`: Run inference and generate summary
- `load_summarizer()`: Convenience function for initialization

**Model Configuration**:
- Base model: TinyLlama-1.1B-Chat-v1.0
- Quantization: 4-bit (NF4)
- Compute dtype: float16
- Device: Auto (CUDA if available, else CPU)

**Prompt Template**:
```
### Instruction:
[System prompt about financial analysis]

### Document:
[Financial report text - truncated to 3000 chars]

### Summary:
[Generated summary appears here]
```

**Generation Parameters**:
- max_new_tokens: 256
- num_beams: 1
- do_sample: False
- temperature: 0.7

---

#### 4. Model Storage (`model/fine_tuned_model.pt`)
**Purpose**: Store fine-tuned model weights

**Contents**:
- Model state dictionary (trained weights)
- Training epoch information
- Validation metrics
- Model configuration

**Size**: 1.02 GB (1,075,755,333 bytes)

**Format**: PyTorch checkpoint (.pt)

---

### Data Flow

```
PDF Upload
    │
    ▼
[Validate PDF]
    │
    ▼
[Extract Text] ──────────────┐
    │                        │
    ▼                        │
[Clean Text]                 │
    │                        │
    ▼                        ▼
[Format Prompt] ──────> [Text Preview]
    │                        │
    ▼                        │
[Tokenize]                   │
    │                        │
    ▼                        │
[Model Inference]            │
    │                        │
    ▼                        │
[Decode Output]              │
    │                        │
    ▼                        ▼
[Extract Summary] ──────> [Display Results]
```

### Technology Stack

**Frontend**:
- Gradio 4.0+ (Web UI framework)

**Backend**:
- Python 3.8+
- PyTorch 2.0+ (Deep learning framework)
- Transformers 4.35+ (HuggingFace library)
- PEFT 0.6+ (Parameter-efficient fine-tuning)
- BitsAndBytes 0.41+ (Quantization)

**Utilities**:
- PyPDF2 3.0+ (PDF processing)
- Accelerate 0.24+ (Model optimization)

### Performance Characteristics

**Model Loading**:
- First run: ~60 seconds (downloads base model)
- Subsequent runs: ~30 seconds

**Inference Time**:
- CPU: 20-60 seconds per document
- GPU (A100): 5-15 seconds per document

**Memory Requirements**:
- CPU: ~4GB RAM minimum
- GPU: ~8GB VRAM recommended

**Document Processing**:
- Small PDFs (<10 pages): 5-10 seconds
- Medium PDFs (10-50 pages): 15-30 seconds
- Large PDFs (>50 pages): 30-60 seconds

### Security Considerations

**Current Implementation**:
- Local execution only
- No authentication required
- No data persistence
- No external API calls (except HuggingFace model download)

**Limitations**:
- Not production-ready
- No rate limiting
- No input sanitization beyond basic validation
- No logging or monitoring

### Scalability Notes

**Current Limitations**:
- Single-threaded processing
- One request at a time
- No queue management
- No caching

**Potential Improvements**:
- Add request queuing
- Implement result caching
- Use async processing
- Deploy with multiple workers
- Add load balancing

### Error Handling

**PDF Processing Errors**:
- Invalid file format → User-friendly error message
- Encrypted PDF → "Cannot process encrypted PDFs"
- No extractable text → "PDF contains no text"

**Model Errors**:
- Model not found → "Model file missing"
- Out of memory → "Insufficient memory"
- CUDA errors → Fallback to CPU

**UI Errors**:
- No file uploaded → "Please upload a file"
- Processing timeout → "Request timed out"
