# CS6111-relation-extraction
Extract SQL relation from plain text using SpanBERT and Generative AI models

# Credentials
- Google API Key : AIzaSyCqNuAyVdEZHSXjsenXecoF3doj9CmIzzE
- Engine ID: a3e855684755e4cb8
# How to Run
First, open our proj2 directory and  follow the instruction on the website: https://www.cs.columbia.edu/~gravano/cs6111/Proj2/ to install the required packages

**After installing all the packages:**
Create `main.py` under the SpanBERT directory, and copy the code in `main.py` under proj2 directory to main.py under the SpanBERT directory.
Moreover, modify `spacy_help_functions.py` under SpanBERT directory using our own written one (which is given under proj2 directory also) 

Then, you can run our main.py under SpanBERT directory, with the following input:
`python3 main.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>`

If there are any missing packages when running, please run pip install to install all other missing packages

# Project Internal Design

## Overview

This project implements an Information Extraction System (ISE) using either the SpanBERT model or Google Gemini API for extracting relations from webpages. The system is designed to query Google's Custom Search Engine for webpages related to a specific query and then process these webpages to extract meaningful relations based on the specified parameters. 

## Main Components

1. **Google Custom Search Integration**: Utilizes Google's Custom Search API to fetch the top-10 URLs related to the input query. This component is responsible for initiating the search and processing the search results.

2. **Webpage Processing**: For each URL, this component retrieves the webpage content, utilizes BeautifulSoup to parse HTML and extract text, and then processes the text to keep only the first 10,000 characters if necessary.

3. **Text Annotation and Entity Recognition**: Utilizes spaCy for natural language processing to annotate the text, split it into sentences, and identify named entities (e.g., PERSON, ORGANIZATION).

4. **Relation Extraction**:
   - **SpanBERT Integration**: For -spanbert mode, uses a pre-trained SpanBERT model to predict relations between named entities identified in the text.
   - **Google Gemini Integration**: For -gemini mode, constructs prompts for Google Gemini based on the sentences and entities, then sends these prompts to the Gemini API for relation extraction.

5. **Relation Processing and Deduplication**: Identifies tuples with a minimum confidence threshold, removes exact duplicates while keeping the highest confidence tuples, and manages the set of extracted tuples (X).

6. **Query Management and Iteration**: Manages the iterative querying process by generating new queries based on the extracted tuples, tracking already used queries, and determining when the process has stalled or when enough tuples have been extracted.

#### External Libraries

- `googleapiclient.discovery`: Used for interacting with Google's Custom Search API.
- `requests`: Used for making HTTP requests to retrieve webpages.
- `BeautifulSoup` from `bs4`: Used for parsing HTML content of webpages and extracting text.
- `spacy`: Used for natural language processing tasks, including text annotation, sentence splitting, and named entity recognition.
- `SpanBERT` from `spanbert`: Used for relation extraction based on the SpanBERT model.
- `google.generativeai`: Used for interacting with Google's Gemini API for relation extraction in -gemini mode.

### Step 3 Execution Details

Step 3 involves annotating webpage text and extracting information based on the specified relation. This step is carried out differently based on the mode (-spanbert or -gemini) specified by the user.

#### In -spanbert Mode

1. **Text Annotation**: Each sentence from the webpage text is processed with spaCy to identify and tag named entities.
2. **Entity Pair Creation**: For each sentence, create pairs of named entities that match the required types for the specified relation.
3. **Relation Extraction with SpanBERT**: For each entity pair, the SpanBERT model predicts the relation between the entities and the confidence score.
4. **Tuple Collection**: Tuples with a confidence score above the threshold are added to the set X. Duplicate tuples are managed as described, keeping only the highest confidence instances.

#### In -gemini Mode

1. **Text Annotation**: Similar to -spanbert mode, sentences are annotated using spaCy for named entities.
2. **Prompt Construction**: For sentences with relevant named entity pairs, construct prompts for Google Gemini that describe the task of extracting the specific relation.
3. **Relation Extraction with Gemini**: Sends the constructed prompts to Google Gemini and processes the responses to extract relations.
4. **Tuple Collection**: All extracted tuples are added to the set X, assuming a hard-coded confidence of 1.0, as Gemini does not provide confidence scores. Duplicate and deduplication management is similar to -spanbert mode.

#### Query Management

After processing webpages for a query, the system evaluates if enough tuples have been extracted. If not, it selects a new query based on the highest confidence tuple from X that hasn't been used for querying yet, and repeats the process. The iteration continues until the desired number of tuples is extracted or the process stalls.
