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
# prep NLTK Stop words
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

base_folder_name = "mmip"
code_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/code"
ranged_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/ranged_data"
section_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/data_sections"
analyzed_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/analyzed_data"
output_folder = 'c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/output/'
output_folder = 'c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/california_output/'
california_folder =  "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/data_california"
california_output = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/california_output"
strick_california_output = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/strick_california_output"
strick_california_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/strik_data_california"
mmip_folder = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/mmip"
mmip_output = "c:/Users/jacks/Documents/INTA 6450 Data Analytics and Security/Project/group/mmip_output"




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
            print(f"finished processing: {i} documents, {str(round(i / 10954.0 * 100 , 2))}% processed")

    write_to_file(base_folder_name + '_body', body, section_folder)
    write_to_file(base_folder_name + '_from_info', email_from, section_folder)
    write_to_file(base_folder_name + '_to_info', email_to, section_folder)
    return {
        'body': body,
        'to': email_from,
        'from_': email_to
    }

#load the data in the section folder
def load_section_data():
    f = open(section_folder + "/" + base_folder_name + "_body.txt")
    body = json.load(f)
    f.close()

    f = open(section_folder + "/" + base_folder_name + "_from_info.txt")
    email_from = json.load(f)
    f.close()

    f = open(section_folder + "/" + base_folder_name + "_to_info.txt")
    email_to = json.load(f)
    f.close()

    return {
        'body': body,
        'to': email_from,
        'from_': email_to
    }


def get_all_data(mypath):
    result = []
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        full_path = os.path.join(mypath, file)
        data = load_data_from_file(full_path)
        data["file_name"] = full_path
        result.append(data)
    return result


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


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def dataframe_to_file(dataframe, folder):
    for i in range(len(dataframe)):
        data={}
        data['keywords'] = dataframe.iloc[i][2]
        data['topic_perc_contribution'] = dataframe.iloc[i][1]
        data['text'] = dataframe.iloc[i][3]
        with open(folder + str(i) + '.json', 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    
    
    # sort and write the data into body, from, to sections
    data = reorg_data_to_section(mmip_folder)

    # Start of analyzation
    data = load_section_data()
    email_df = pd.DataFrame(data)





    # prep NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])



    # Convert email body to list
    data = email_df.body.values.tolist()
    data_words = list(sent_to_words(data)) 



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
    write_to_file(base_folder_name + '_words_nostops', data_words_nostops, analyzed_folder)
    write_to_file(base_folder_name + '_words_bigrams', data_words_bigrams, analyzed_folder)
    

    # Remove Stop Words
    data_words_nostops = load_data_from_file(analyzed_folder + "/" + base_folder_name + "_words_nostops.txt")
    
    # Form Bigrams
    data_words_bigrams = load_data_from_file(analyzed_folder + "/" + base_folder_name + "_words_bigrams.txt")


    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, ['NOUN', 'ADJ', 'VERB', 'ADV'], nlp)
    write_to_file(base_folder_name + '_lemmatized', data_lemmatized, analyzed_folder)
    #print(data_lemmatized[200])
    
    data_lemmatized = load_data_from_file(analyzed_folder + "/" + base_folder_name + "_lemmatized.txt")

    # create dictionary and corpus both are needed for (LDA) topic modeling

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    
    mallet_path = 'C:/mallet/mallet-2.0.8/bin/mallet'
    os.environ['MALLET_PATH'] = 'C:/mallet/mallet-2.0.8/'
    os.environ['MALLET_HOME'] = 'C:/mallet/mallet-2.0.8/'


    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                corpus=corpus, 
                                                num_topics=20, 
                                                id2word=id2word)

    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
    

    # Show graph
    limit=40 
    start=2 
    step=6
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


    # Select the model and print the topics
    optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=32, id2word=id2word)
    model_topics = optimal_model.show_topics(formatted=False)
    print(optimal_model.print_topics(num_words=10))

    optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=32, id2word=id2word)
    coherencemodel = CoherenceModel(model=optimal_model, texts=texts, dictionary=id2word, coherence='c_v')
    model_topics = optimal_model.show_topics(formatted=False)
    print(optimal_model.print_topics(num_words=10))


    data = load_section_data()
    email_df = pd.DataFrame(data)


    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Convert email body to list
    data = email_df.body.values.tolist()

    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']



    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    #sent_topics_sorteddf_mallet.to_csv('california_output.csv', sep='\t', encoding='utf-8')
    dataframe_to_file(sent_topics_sorteddf_mallet, mmip_output)
    

    #strick_california_output
    #california_output
    
    '''
    pick = 5
    switch = True
    parsed_data = get_all_data(strick_california_output)
    i = 0
    for file in parsed_data:
        print(str(i) + " : " + file['keywords'] + ", topic contribution: " + str(file["topic_perc_contribution"]))
        #print("topic contribution: " + str(file["topic_perc_contribution"]))
        #print("text: " + file["text"][0:100])
        i+=1
    '''