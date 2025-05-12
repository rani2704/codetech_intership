import nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Force NLTK to look in the correct folder
nltk.data.path.append(r'C:\Users\HP\AppData\Roaming\nltk_data')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

#nltk.download('punkt', download_dir=r'C:\Users\HP\AppData\Roaming\nltk_data')
#nltk.data.path.append(r'C:\Users\HP\AppData\Roaming\nltk_data')
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize, sent_tokenize
#import string

# Download resources (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, max_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Create a frequency table for words
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    words = word_tokenize(text.lower())
    freq_table = {}

    for word in words:
        if word not in stop_words:
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

    # Score each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                if sentence in sentence_scores:
                    sentence_scores[sentence] += freq_table[word]
                else:
                    sentence_scores[sentence] = freq_table[word]

    # Sort sentences by score and select top ones
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    summary = ' '.join(summarized_sentences)
    return summary

# Example usage
if __name__ == "__main__":
    sample_text = """
    Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
    The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
    The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal.
    AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary approach based on mathematics, computer science, linguistics, psychology, and more.
    """

    summary = summarize_text(sample_text)
    print("Original Text:\n", sample_text)
    print("\nSummarized Text:\n", summary)
