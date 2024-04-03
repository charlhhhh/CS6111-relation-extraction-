import itertools
import math
import os
import re
import string
import sys
import time

import requests
import spacy
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from spacy_help_functions import extract_relations
from spanbert import SpanBERT
import google.generativeai as genai

# To pair the relation number to the real relation
relation_pair = {1: 'Schools_Attended', 2: 'Work_For', 3: 'Live_In', 4: 'Top_Member_Employees'}

# Load spaCy model for English language
nlp = spacy.load("en_core_web_lg")

# Initialize SpanBERT
spanbert = SpanBERT("./pretrained_spanbert")  

def get_gemini_completion(prompt, model_name, max_tokens, temperature, top_p, top_k):

    # Initialize a generative model
    model = genai.GenerativeModel(model_name)

    # Configure the model with your desired parameters
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Generate a response
    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text
# Get the google custom search results
def google_custom_search(google_api_key, google_engine_id, query):
    service = build("customsearch", "v1", developerKey=google_api_key)
    res = service.cse().list(q=query, cx=google_engine_id).execute()
    return res

# Function to print the initial information
def print_initial_info(google_api_key, google_engine_id, google_gemini_key, method, relation_number, threshold, query, k, ir_number):
    print('Parameters:')
    print(f'Client key  = {google_api_key}')
    print(f'Engine key  = {google_engine_id}')
    print(f'Gemini key  = {google_gemini_key}')
    print(f"Method  = {method[1:]}")
    print(f"Relation        = {relation_pair[relation_number]}")
    print(f"Threshold       = {threshold}")
    print(f"Query           = {query}")
    print(f"# of Tuples     = {k}")
    print("Loading necessary libraries; This should take a minute or so ...")
    print(f'======Iteration: {ir_number} - Query: {query}================')

def print_extracted_relations(sorted_relations, relation_name, ir_number):
    num_relations = len(sorted_relations)

    print(f"================== ALL RELATIONS for {relation_name} ( {num_relations} ) =================")
    for relation, confidence in sorted_relations:
        print(f"Confidence: {confidence:.8f} \t| Subject: {relation[0]} \t| Object: {relation[2]}")
    print(f"Total # of iterations = {ir_number + 1}")
def print_extr_gem(sorted_relations, relation_name, ir_number):
    num_relations = len(sorted_relations)
    print(f"================== ALL RELATIONS for {relation_name} ( {num_relations} ) =================")
    for relation, confidence in sorted_relations:
        print(f"Subject: {relation[0]} \t| Object: {relation[1]}")
    print(f"Total # of iterations = {ir_number + 1}")



# For step 3.a - 3.c: Fetch and process the webpage 
def fetch_and_process_webpage(url):
    try:
        response = requests.get(url, timeout=5) 
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        
        if len(text) > 10000: 
            print(f"Trimming webpage content from {len(text)} to 10000 characters") 
            print("Webpage length (num characters): 10000")
            text = text[:10000]
        else:
            print(f"Webpage length (num characters): {len(text)}") 
        text = re.sub('\n+', ' ', text)
        text = re.sub(u'\xa0', ' ', text) 
        return text
    except requests.exceptions.RequestException as e: 
        print(f"Error fetching URL {url}: {str(e)}")
        return None
    except Exception as e: 
        print(f"Unexpected error when processing URL {url}: {str(e)}")
        return None

# For Spanbert, 3.d - 3.f
def process_text(text, sub, obj, nlp, spanbert, internal_name, threshold, k, relation_number, X_confidence, gemini=False):
    doc = nlp(text)
    sentences = list(doc.sents)
    print(f'Extracted {len(sentences)} sentences. Processing each sentence one by one to check for the presence of right pair of named entity types; if so, will run the second pipeline...')
    
    # number of annotated sentences
    #num_annotated = 0
    # number of sentence that the confidence is larger than threshold
    #num_confidence = 0
    # Total sentences extracted
    #num_extracted = 0

    stats = {'num_extracted': 0, 'num_confidence': 0, 'num_annotated': 0}

    # entities_of_interest
    if obj != 'GPE':
        entities_of_interest = [sub] + (obj if isinstance(obj, list) else [obj])
    else:
        entities_of_interest = ['PERSON', 'LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY'] 
    # print(entities_of_interest)
    if sub == 'ORGANIZATION':
        sub = ['ORG', 'ORGANIZATION']
    if obj == 'ORGANIZATION':
        obj = ['ORG', 'ORGANIZATION']

    #X_confidence = {}  

    for i, sentence in enumerate(doc.sents):
        if i % 5 == 0 and i != 0:
            print(f'Processed {i} / {len(sentences)} sentences')

        entities = [ent for ent in sentence.ents]
        #print(entities)

        #entity_pairs = []
        found = False
        sent = nlp(sentence.text)
        for i, entity1 in enumerate(entities):
            if found:
                break
            for j, entity2 in enumerate(entities):
                #print(entity1.label_)
                #print(entity2.label_)
                #print(sub)
                #print(obj)
                condition = entity1.label_ in sub and entity2.label_ in obj
                if condition:
                    found = True
                    #entity_pairs.add((entity1, entity2))
        # print(entity_pairs)
        
        if found:
               # if entity1 != entity2 and (entity1.label_ == sub and entity2.label_ in obj):
            # print("1111111")
            if not gemini: # use spanbert
                try:
                    # relations = False
                    relations = extract_relations(sent, spanbert, internal_name, entities_of_interest, threshold, stats, X_confidence)
                    # print("Relations extracted");
                    if relations:
                        #num_annotated += 1
                        #print("222222")
                        for relation, conf in relations.items():
                            #num_extracted += 1
                            #print(relation)
                            #if conf >= threshold:
                                #num_confidence += 1
                            if conf > X_confidence.get(relation, 0):
                                X_confidence[relation] = conf
                except IndexError:  
                    pass
            else:
                if relation_number == 1:
                    rel = "school attended"
                elif relation_number == 2:
                    rel = "work for or worked for"
                elif relation_number == 3:
                    rel = "live in or lived in"
                    obj = "location" # simplify
                elif relation_number == 4:
                    rel = "top Member employees"

                prompt = f"""Given a sentence, extract tuples of entities that satisfy relation: {rel}.
                Return None if no such tuple can be found. Do not include brackets. Separate tuples with semicolon.
                Order it as: {sub}, {obj}
                Example: from sentence "John works for Apple", if the relation needed is work for, return this: John, Apple
                from sentence "John lives in London and Mary lives in Beijing", if the relation needed is live in, 
                return this: John, London; Mary, Beijing.
                Always make sure if the output is not "None", there exist at least one comma in the output.
                sentence: {sentence.text} 
                extracted:"""
                time.sleep(0.2)
                response_text = get_gemini_completion(prompt, 'gemini-pro', 100, 0.2, 1, 32)
                if response_text != 'None':
                    
                    if ';' in response_text: # extract list of tuples
                        stats['num_annotated'] += 1
                        resp_ = response_text.split("; ")
                        for tup in resp_: #for each tuple:
                            resp = tup.split(", ")
                            stats['num_extracted'] += 1
                            print("\n\t\t=== Extracted Relation ===")
                            print(f"\t\tSentence: {sentence.text}")
                            print(f"\t\tSubject: {resp[0]}; Object: {resp[1]};")
                            relation = (resp[0].lower(),resp[1].lower())
                            if relation in X_confidence: # duplicate
                                print("\t\tDuplicate. Ignoring this.")
                                continue
                            else: # add to set
                                print("\t\tAdding to set of extracted relations")
                                
                                stats['num_confidence'] += 1                    
                                X_confidence[relation] = 1.0
                    else:
                        resp = response_text.split(", ")
                        
                        stats['num_extracted'] += 1
                        print("\n\t\t=== Extracted Relation ===")
                        print(f"\t\tSentence: {sentence.text}")
                        print(f"\t\tSubject: {resp[0]}; Object: {resp[1]};")
                        relation = (resp[0].lower(),resp[1].lower())
                        if relation in X_confidence: # duplicate
                            print("\t\tDuplicate. Ignoring this.")
                            continue
                        else: # add to set
                            print("\t\tAdding to set of extracted relations")
                            
                            stats['num_confidence'] += 1                    
                            X_confidence[relation] = 1.0
                    print("\t\t==========")
    
    print(f"Extracted annotations for {stats['num_annotated']} out of total {len(sentences)} sentences")
    print(f"Relations extracted from this website: {stats['num_confidence']} (Overall: {stats['num_extracted']})")
    #print(X_confidence)
    return X_confidence

def remove_duplicates_and_sort(X_confidence, k):
    # Remove duplicates and keep the one with the highest confidence
    unique_relations = {}
    for relation, conf in X_confidence.items():
        if relation not in unique_relations or conf > unique_relations[relation][1]:
            unique_relations[relation] = (relation, conf)
    
    # Sort by confidence in descending order
    sorted_relations = sorted(unique_relations.values(), key=lambda x: x[1], reverse=True)
    
    # Return top-k if possible
    return sorted_relations[:k] if len(sorted_relations) >= k else sorted_relations


def main():
    # python3 project2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>
    
    # To correct the argument -> need revision for this part
    #if not (sys.argv[3].replace('.', '', 1).isdigit() and len(sys.argv) == 5):
    #    sys.exit(f"Usage: {sys.argv[0]} [Google API Key] [Google Engine ID] [Precision] [Query]")

    # Set the initial values
    method = sys.argv[1]
    google_api_key = sys.argv[2]
    google_engine_id = sys.argv[3]
    google_gemini_key = sys.argv[4]
    relation_number = int(sys.argv[5])
    threshold = float(sys.argv[6])
    seed_query = " ".join(sys.argv[7: len(sys.argv)-1]) # plausible relation tuple e.g."bill gates microsoft"
    k = int(sys.argv[len(sys.argv)-1])
    ir_number = 0
    genai.configure(api_key=google_gemini_key)
    # To remove the double quotes of seed query
    query = seed_query.strip('\'\"”“')

    # To set the relation
    if relation_number == 1:
        sub = 'PERSON'
        obj = 'ORGANIZATION'
        internal_name = 'per:schools_attended'
    elif relation_number == 2:
        sub = 'PERSON'
        obj = 'ORGANIZATION'
        internal_name = 'per:employee_of'
    elif relation_number == 3:
        sub = 'PERSON'
        obj = 'GPE'
        internal_name = 'per:cities_of_residence'
    elif relation_number == 4:
        obj = 'PERSON'
        sub = 'ORGANIZATION'
        internal_name = 'org:top_members/employees'

    # Print initial information
    print_initial_info(google_api_key, google_engine_id, google_gemini_key, method, relation_number, threshold, query, k, ir_number)

    
    # initialize extracted tuples X
    X = set()
    X_confidence = {}
    queried_tuples = set()
    queried_tuples.add(query)
    #print(queried_tuples)

    # Number of tuples retrived
    num_of_tuples = 0

    while True:
        results = google_custom_search(google_api_key, google_engine_id, query)

        for i, item in enumerate(results['items'][:10]):
            url = item['link']
            print(f'URL ({i + 1}/10): {url}')
            print("Fetching text from url...")
            text = fetch_and_process_webpage(url)
            print('Annotating the webpage using spacy')
            if text:
                if method == '-spanbert':
                    results = process_text(text, sub, obj, nlp, spanbert, internal_name, threshold, k, relation_number, X_confidence)
                else:
                    results = process_text(text, sub, obj, nlp, spanbert, internal_name, threshold, k, relation_number, X_confidence, gemini=True)
                X_confidence.update(results)  # Update the main X_confidence dictionary with new values

        top_k_relations = remove_duplicates_and_sort(X_confidence, k)
        if method == '-spanbert':
            print_extracted_relations(top_k_relations, internal_name, ir_number)
        else:
            print_extr_gem(top_k_relations, internal_name, ir_number)
        ir_number += 1
        if len(top_k_relations) >= k:
            # print(f"Found {len(top_k_relations)} relations")
            break  
        
        new_query_found = False
        for relation, confidence in top_k_relations:
            if method == '-spanbert':
                subject, _, object_ = relation
                q = subject + ' ' + object_
            else:
                subject, object_ = relation
                q = subject + ' ' + object_
            if q not in queried_tuples:
                
                query = q
                queried_tuples.add(q)  
                new_query_found = True
                print(f'======Iteration: {ir_number} - Query: {query}================')
                break
        if not new_query_found:
            print("No new query can be generated. Stopping.")
            break 



if __name__ == "__main__":
    main()



