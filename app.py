import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# FOR TEXT PROCESSING
import re
from unidecode import unidecode # to conver special characters to their equivalents
# FOR QUESTION-ANSWER PAIR GENERATION
import pandas as pd
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_wiki_question_answer(url):
    # Parsing the data for  creating BeautifulSoup object

    # Get the HTML of the Wikipedia Page
    page = requests.get(url)

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page.text,'html.parser')

    # A list of <h2> tags that I deem unnecessary
    exclude_tags = soup.find_all('h2', text=["See also", 
                                             "Explanatory notes",
                                            "References",
                                            "External links",
                                            "Further reading"])
    # String to store all content from the <p> tags
    text = []

    # BeautifulSoup function that finds all the <p> tags
    p_tags = soup.find_all('p')

    # Loop through the important <p> tags and add the content from it into the text variable
    for paragraph in p_tags:
        parent = paragraph.find_parent('h2')
        if parent and parent in exclude_tags:
            # Skip paragraphs that are children of excluded sections
            continue
        text.append(paragraph.text)
    # the deprecated warning is not relevant, https://pyup.io/packages/pypi/beautifulsoup4/changelog
    
    final_text = ""
    special_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '...']
    pattern = '|'.join(re.escape(char) for char in special_chars) # creating a pattern for the re.sub() function
    # Process paragraphs
    for paragraph in text:
        paragraph = re.sub(r'\[[0-9a-zA-Z]*\]','',paragraph) # for citations
        paragraph = re.sub(pattern,'',paragraph) # for special characters
        paragraph = re.sub(r"\n", " ", paragraph) # for extra lines 
        paragraph = re.sub(r"\s+", " ", paragraph) # for extra white spaces
        paragraph = paragraph.lower()
        paragraph = unidecode(paragraph)
        final_text += paragraph + "\n" # combining it into one string
    
    # Load the T5 tokenizer and pre-trained model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    # Process text into sentences, avoiding splitting cases such as 2.0, Mr. etc
    sentences = re.split(r'(?<!\d)[.!?]', final_text)

    # Generate question-answer pairs
    qa_pairs = []
    for sentence in sentences:
        if len(sentence) > 10: # ignore short sentences
            input_text = "generate questions: " + sentence.strip()
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids=input_ids, max_length=60, num_beams=5, early_stopping=True)
            question = tokenizer.decode(outputs[0]).replace("question: ", "").replace("<pad> ", "").replace("</s>", "")
            qa_pairs.append({"question": question, "answer": sentence.strip()})
    df = pd.DataFrame(qa_pairs, columns=["question", "answer"])

    df.to_csv('output.csv', index=False)
    return df

# Create ST App 

st.title("Wikipedia Question-Answer Generator")

# Ask user for Wikipedia URL
wiki_url = st.text_input("Enter a Wikipedia URL")

# Generate questions and answers
if st.button("Generate"):
    if wiki_url:
        with st.spinner("This might take some time please be patient..."):
            df = generate_wiki_question_answer(wiki_url)
        for i, row in df.iterrows():
            with st.expander(row['question']):
                st.write(row['answer'])
    else:
        st.write("Please enter a Wikipedia URL.")



