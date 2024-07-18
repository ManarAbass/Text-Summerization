# NLP Packages
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Packages for Normalizing Text
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

# Import heapq for Finding the Top N Sentences
from heapq import nlargest

def text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    
    # Build Word Frequency
    word_frequencies = {}
    for word in docx:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text.lower() not in word_frequencies.keys():
                word_frequencies[word.text.lower()] = 1
            else:
                word_frequencies[word.text.lower()] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequency

    # Sentence Tokens
    sentence_list = [sentence for sentence in docx.sents]

    # Sentence Scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    summarized_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary

# Test the summarizer
print(text_summarizer("We are currently experiencing another gold rush in AI. Billions are being invested in AI startups across every imaginable industry and business function. Google, Amazon, Microsoft and IBM are in a heavyweight fight investing over $20 billion in AI in 2016. Corporates are scrambling to ensure they realise the productivity benefits of AI ahead of their competitors while looking over their shoulders at the startups. China is putting its considerable weight behind AI and the European Union is talking about a $22 billion AI investment as it fears losing ground to China and the US."))
