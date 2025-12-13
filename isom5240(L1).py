# Import part
# use a pipeline as a high-level helper
# use docx for demo
!pip install python-docx
from transformers import pipeline, AutoTokenizer
import docx

# Function part
# part1: text_summarization
def text_summarization(file_path):
    # Manually load and configure the tokenizer to ensure input truncation
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.model_max_length = 1024 # Set the maximum input length for the tokenizer

    # Initialize pipeline with the configured tokenizer
    text_summarization_model1 = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer)

    # Read the content of the docx file
    document = docx.Document(file_path)
    text_content = "\n".join([paragraph.text for paragraph in document.paragraphs])

    input_text = text_summarization_model1(
        text_content,
        max_length=150, # Set a reasonable max length for the generated summary (output)
        min_length=50,  # Set a reasonable min length for the generated summary (output)
        truncation=True # Explicitly truncate input using tokenizer.model_max_length
    )[0]["summary_text"]
    return input_text

# Main part
print("Title：Robo-Advisor")

# Image Input
summarization_name = input("Enter an text file name: ")
text_description = text_summarization(summarization_name)
print(f"Generated text: {text_description}")
