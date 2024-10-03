from transformers import BartTokenizer,pipeline
from transformers import PipelineException

tokenizer=BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer=pipeline("summarization",model="facebook/bart-large-cnn")
def summarize_text(text,max_length=130,min_length=30):
    try:
        if len(text.strip())==0:
            raise ValueError("Input text is empty.")
        
        tokenized_inputs=tokenizer(text,return_tensors="pt",truncation=True,max_length=1024)
        if len(tokenized_inputs['input_ids'][0])>1024:
            print("Warning: Input text is too long and has been truncated to fit the model's maximum token length.")
        summary=summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=6,
            length_penalty=2.0,
            early_stopping=True
        )
    
        return summary[0]['summary_text']
    
    except ValueError as ve:
        return (f"Error: {ve}")
    except PipelineException as pe:
        return(f"PipelineException: {pe}")
    except Exception as e:
        return(f"An unexpected error occurred: {e}")

user_text=input("Enter the text you want to summarize: ")
try:
    max_len=int(input("Enter the maximum summary length (default 130): "))
    min_len=int(input("Enter minimum summary length (defult 30): "))
except ValueError:
    print("Invalid input for length, using default values.")
    max_len=130
    min_len=30

summary=summarize_text(user_text,max_length=max_len,min_length=min_len)
if summary:
    print("Summarized Text: ",summary)