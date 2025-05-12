from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(input_text):
    
    summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


if __name__ == "__main__":

    user_input = input("Enter the text you want to summarize: ")
    
    
    summary = summarize_text(user_input)
    
    
    print("\nSummary:")
    print(summary)
