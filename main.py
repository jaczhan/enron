import json
import os
import shutil
import pandas as pd
import re
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords

# spacy for lemmatization
import spacy

# for plotting
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from datetime import datetime


code_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/code"
ranged_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/ranged_data"
section_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/data_sections"
analyzed_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/analyzed_data"


def filter_files(mypath):
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    d1 = datetime(2000, 1, 1)
    d2 = datetime(2001, 9, 30)
    for file in files:
        full_path = os.path.join(mypath, file)
        f = open(full_path)
        data = json.load(f)

        date = data['date']
        date_datetime = datetime.fromisoformat(date[:-5])

        if date_datetime >= d1 and date_datetime <= d2:
            shutil.copy2(full_path, "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/ranged_data")


def write_to_file(file_name, data, folder):

    path = os.path.join(folder, file_name)
    path += ".txt"
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


# write the data into sections of body/from/to
def reorg_data_to_section(mypath):
    body=[]
    email_from=[]
    email_to=[]

    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    i = 0
    for file in files:
        i+=1
        full_path = os.path.join(mypath, file)
        f = open(full_path)
        data = json.load(f)

        body.append(data['text'])
        if 'from' in data:
            email_from.append(data['from'][0]['address'])
        else:
            email_from.append("none")

        if 'to' in data:
            email_to.append(data['to'][0]['address'])
        else:
            email_to.append("none")

        if i % 1000 == 0:
            print(f"finished processing: {i} documents, {str(round(i / 158739.0 * 100 , 2))}% processed")

    write_to_file('body', body, section_folder)
    write_to_file('from_info', email_from, section_folder)
    write_to_file('to_info', email_to, section_folder)
    return {
        'body': body,
        'to': email_from,
        'from_': email_to
    }

#load the data in the section folder
def load_section_data():
    f = open(section_folder + "/body.txt")
    body = json.load(f)
    f.close()

    f = open(section_folder + "/from_info.txt")
    email_from = json.load(f)
    f.close()

    f = open(section_folder + "/to_info.txt")
    email_to = json.load(f)
    f.close()

    return {
        'body': body,
        'to': email_from,
        'from_': email_to
    }


def load_data_from_file(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data


# remove stop_words, make bigrams and lemmatize
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags, nlp):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# tokenize - break down each sentence into a list of words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


if __name__ == '__main__':
    '''
    # only use data in the right time range that we are interested in
    file_path = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/dataset"
    filter_files(file_path)
    

    # sort and write the data into body, from, to sections
    data = reorg_data_to_section(ranged_folder)
    '''


    '''
    # Start of analyzation

    data = load_section_data()
    email_df = pd.DataFrame(data)
    print(email_df.head())


    # prep NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    print(email_df.iloc[2]['body']) # displays info below


    # Convert email body to list
    data = email_df.body.values.tolist()
    data_words = list(sent_to_words(data)) 
    print(data_words[3])


    from gensim.models.phrases import Phrases, Phraser
    # Build the bigram and trigram models
    bigram = Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)


    # save the computed data for faster future processing
    write_to_file('data_words_nostops', data_words_nostops, analyzed_folder)
    write_to_file('data_words_bigrams', data_words_bigrams, analyzed_folder)
    

    # Remove Stop Words
    data_words_nostops = load_data_from_file(analyzed_folder + "/data_words_nostops.txt")
    
    # Form Bigrams
    data_words_bigrams = load_data_from_file(analyzed_folder + "/data_words_bigrams.txt")


    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, ['NOUN', 'ADJ', 'VERB', 'ADV'], nlp)
    write_to_file('data_lemmatized', data_lemmatized, analyzed_folder)
    print(data_lemmatized[200])
    '''
    data_lemmatized = load_data_from_file(analyzed_folder + "/data_lemmatized.txt")

    # create dictionary and corpus both are needed for (LDA) topic modeling

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]



    
    mallet_path = code_folder + '/mallet-2.0.8/bin/mallet'
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
    print(lda_model.print_topics())# The weights reflect how important a keyword is to that topic.

    doc_lda = lda_model[corpus]


    # Model perplexity and topic coherence provide a convenient
    # measure to judge how good a given topic model is.
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


    # Visualize the topics
    pyLDAvis.enable_notebook(sort=True)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.display(vis)


