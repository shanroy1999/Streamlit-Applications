import streamlit as st
import spacy
from textblob import TextBlob
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def text_analyzer(text):
    nlp = spacy.load('en')
    doc = nlp(text)

    # Tokenize the text
    tokens = [token.text for token in doc]
    data = [('"Tokens":{},\n"Lemma": {}'.format(token.text, token.lemma_)) for token in doc]
    return data

def entity_extractor(text):
    nlp = spacy.load('en')
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def sumy_summarizer(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def main():
    st.title("NLP Application made with Streamlit")
    st.subheader("Natural Language Processing on the Go")
    message = st.text_area("Enter Your Text here", "Type Here")

    # Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize your Text")
        if st.button("Tokenize"):
            result = text_analyzer(message)
            st.json(result)

    # Named Entity
    if st.checkbox("Show Named Entity"):
        st.subheader("Extract Entities from your Text")
        if st.button("Extract Entities"):
            result = entity_extractor(message)
            st.json(result)

    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Sentiments of your Text")
        if st.button("Analyze Sentiments"):
            blob = TextBlob(message)
            result = blob.sentiment
            st.success(result)

    # Summarize Text
    if st.checkbox("Summarize the Text"):
        st.subheader("Summary")
        options = st.selectbox("Choose of your Summarizer", ("gensim", "sumy"))
        if st.button("Summarize"):
            if options=="gensim":
                st.text("Using Gensim Summarizer.....")
                result = summarize(message)
            elif options=="sumy":
                st.text("Using Sumy Summarizer.....")
                result = sumy_summarizer(message)
            else:
                str.warning("Using Default Summarizer")
                st.text("Using Gensim")
                result = summarize(message)

            st.success(result)

    st.markdown("<h5 style='text-align: center; color: red;'>Made By Shantanu Roy</h5>", unsafe_allow_html=True)

if __name__=="__main__":
    main()
