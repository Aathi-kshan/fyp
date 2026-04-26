"""
Gradio UI for Financial Report Summarization System.
Provides a simple interface to upload PDFs and generate summaries.
"""

import gradio as gr
import os
import sys

# Add parent directory to path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_loader import extract_text_from_pdf, validate_pdf
from utils.inference import load_summarizer


# Global variable to store the loaded model
summarizer = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "fine_tuned_model.pt")


def initialize_model():
    """
    Initialize the summarization model on startup.
    """
    global summarizer
    
    if not os.path.exists(MODEL_PATH):
        return f"Error: Model file not found at {MODEL_PATH}"
    
    try:
        print("Initializing model... This may take a few moments.")
        summarizer = load_summarizer(MODEL_PATH)
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def process_input_and_summarize(pdf_file, text_input):
    """
    Main function to process PDF or text input and generate summary.
    
    Args:
        pdf_file: Uploaded PDF file from Gradio interface (optional)
        text_input: Direct text input from user (optional)
        
    Returns:
        string: generated_summary (or error message)
    """
    global summarizer
    
    # Check if model is loaded
    if summarizer is None:
        return "Error: Model not initialized. Please restart the application."
    
    # Validate input
    if pdf_file is None and not text_input:
        return "Error: Please either upload a PDF file or enter text to summarize."
    
    if pdf_file is not None and text_input:
        return "Error: Please use either PDF upload OR text input, not both."
    
    try:
        if pdf_file is not None:
            # Process PDF
            pdf_path = pdf_file.name
            print(f"Processing PDF file: {pdf_path}")
            
            # Validate PDF
            if not validate_pdf(pdf_path):
                return f"Error: Invalid PDF file. File: {pdf_path}\n\nPlease ensure:\n- The file is a valid PDF\n- The file is not password-protected\n- The file contains extractable text"
            
            # Extract text from PDF
            print(f"Extracting text from: {pdf_path}")
            extracted_text = extract_text_from_pdf(pdf_path)
            
            # Check if text was extracted
            if not extracted_text or len(extracted_text.strip()) < 50:
                return f"Error: Could not extract sufficient text from PDF. Extracted {len(extracted_text)} characters.\n\nThe PDF may be:\n- Scanned images (no text)\n- Password-protected\n- Corrupted"
            
            print(f"✅ Successfully extracted {len(extracted_text)} characters from PDF.")
            document_text = extracted_text
            
        else:
            # Process text input
            document_text = text_input.strip()
            if len(document_text) < 50:
                return f"Error: Text input is too short. Please provide at least 50 characters to summarize."
            
            print(f"✅ Processing text input ({len(document_text)} characters)")
        
        # Generate summary
        print("Generating summary...")
        summary = summarizer.generate_summary(document_text)
        
        return summary
        
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}\n\nPlease try again with different input."
        print(error_msg)
        return error_msg


def create_ui():
    """
    Create and configure the Gradio interface.
    """
    
    # Custom CSS for minimal styling (optional)
    css = """
    .container {
        max-width: 900px;
        margin: auto;
    }
    """
    
    with gr.Blocks(css=css, title="Financial Report Summarizer") as app:
        
        # Header
        gr.Markdown("# Financial Report Summarization System")
        gr.Markdown("Upload a financial report PDF to generate an automated summary.")
        
        # Model status indicator
        with gr.Row():
            status_text = gr.Textbox(
                label="Model Status",
                value="Initializing model...",
                interactive=False
            )
        
        # Main interface
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Option 1: Upload PDF File")
                # PDF upload component
                pdf_input = gr.File(
                    label="Upload Financial Report (PDF)",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
            with gr.Column():
                gr.Markdown("### Option 2: Enter Text Directly")
                # Text input component
                text_input = gr.Textbox(
                    label="Enter Text to Summarize",
                    lines=8,
                    placeholder="Paste or type your financial report text here (minimum 50 characters)..."
                )
        
        with gr.Row():
            # Submit button (centered)
            submit_btn = gr.Button("Generate Summary", variant="primary", size="lg")
        
        # Summary output (full width)
        with gr.Row():
            summary_output = gr.Textbox(
                label="Generated Summary",
                lines=12,
                interactive=False,
                placeholder="Upload a PDF or enter text, then click 'Generate Summary' to see the AI-generated summary here..."
            )
        
        # Instructions
        gr.Markdown("""
        ### Instructions:
        **Option 1 - PDF Upload:**
        1. Click "Upload Financial Report (PDF)" to select a PDF file
        2. Leave the text input box empty
        
        **Option 2 - Text Input:**
        1. Leave the PDF upload empty
        2. Type or paste your text in the text input box (minimum 50 characters)
        
        **Final Step:**
        3. Click "Generate Summary" to process your input
        4. View the AI-generated summary below
        
        **Note:** Use either PDF upload OR text input, not both. Processing may take a few moments.
        """)
        
        # Event handlers
        submit_btn.click(
            fn=process_input_and_summarize,
            inputs=[pdf_input, text_input],
            outputs=[summary_output]
        )
        
        # Initialize model on load
        app.load(
            fn=initialize_model,
            outputs=[status_text]
        )
    
    return app


def main():
    """
    Main entry point for the application.
    """
    print("="*60)
    print("Financial Report Summarization System")
    print("="*60)
    print(f"Model path: {MODEL_PATH}")
    print("Starting Gradio interface...")
    print("="*60)
    
    # Create and launch the UI
    app = create_ui()
    
    # Launch with public sharing disabled by default
    # Set share=True to create a public link (useful for Colab)
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
