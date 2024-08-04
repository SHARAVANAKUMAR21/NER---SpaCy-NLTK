import spacy
import nltk
import requests
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

# Fetch a news article using News API
def fetch_news_article(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data["status"] != "ok" or len(data["articles"]) == 0:
        return None
    
    article = data["articles"][0]
    title = article.get("title", "")
    description = article.get("description", "")
    
    # Ensure title and description are not None
    title = title if title is not None else ""
    description = description if description is not None else ""
    
    return title + " " + description

# Extract entities using SpaCy
def extract_entities_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Extract entities using NLTK
def extract_entities_nltk(text):
    def get_continuous_chunks(text):
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        continuous_chunk = []
        current_chunk = []

        for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    named_entities = get_continuous_chunks(text)
    return named_entities

# Main function
def main():
    api_key = "d8927d7bd62749f6b49c1cbb4c3bb176"  # Replace this with your actual API key
    article = fetch_news_article(api_key)
    if article:
        print("Article Content:\n", article)
        
        print("\nSpaCy Named Entities:")
        spacy_entities = extract_entities_spacy(article)
        for entity in spacy_entities:
            print(entity)
        
        print("\nNLTK Named Entities:")
        nltk_entities = extract_entities_nltk(article)
        for entity in nltk_entities:
            print(entity)
    else:
        print("Failed to fetch the news article")

if __name__ == "__main__":
    main()
