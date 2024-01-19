from flask import Flask, redirect, render_template, request, jsonify
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from string import ascii_lowercase
from gensim.models import Doc2Vec
import snowballstemmer, re
import requests
from bs4 import BeautifulSoup
import re, sys
from tensorflow.keras.models import load_model
import joblib

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36, Opera/9.80 (Windows NT 6.1; WOW64) Presto/2.12.388 Version/12.18'
}

app = Flask(__name__)

def getsoup(url):
    response = requests.get(url, headers=headers)
    Status_Code = response.status_code
    print(url)
    print(Status_Code)
    
    if Status_Code == 200:
      soup = BeautifulSoup(response.content, features="lxml")
    else:
      soup = getsoup(url)
    return soup 

def getLastPageNumber(soup, site):
    pageNumber = []
    if site == 'flipkart':
        review_number = int(soup.find("span", "_2_R_DZ").text.strip().replace(',', '').split()[-2])
        if review_number <=10:
            lastPage = 1
        else:
            link = soup.find(attrs={"class": "_2MImiq _1Qnn1K"})
            pageNumber = link.find('span').text.strip().replace(',', '').split()
            lastPage1 = pageNumber[len(pageNumber)-1]
            lastPage = int(lastPage1)
    elif site == 'amazon':
        review_number = int(soup.find("div", {"data-hook": "cr-filter-info-review-rating-count"}).text.strip().replace(',', '').split()[-3])
        if review_number <=10:
            lastPage = 1
        else:
            lastPage = review_number // 10
    if lastPage > 500:
        lastPage = 2
    return lastPage


def geturllist(url, lastPage):
    urllistPages = []
    url = url[:-1]
    for i in range(1,lastPage+1):
      urllistPages.append (url + str(i))
    return urllistPages


def getReviews(soup, site, url):
    if site == 'flipkart':
        #Extracting the Titles
        title_sec = soup.find_all("p",'_2-N8zT')
        title = []
        for s in title_sec:
            title.append(s.text)

        #Extracting the Author names
        author_sec = soup.find_all("p","_2sc7ZR _2V5EHH")
        author = []
        for r in author_sec:
            author.append(r.text)

        #Extracting the Text
        Review_text_sec = soup.find_all("div",'t-ZTKy')
        text = []
        for t in Review_text_sec:
            text.append(t.text)
            
        #Extracting the Star rating  
        Rating = soup.find_all("div", {"class": ["_3LWZlK _1BLPMq", "_3LWZlK _32lA32 _1BLPMq", "_3LWZlK _1rdVr6 _1BLPMq"]})    
        rate = []
        for d in Rating:
            rate.append(d.text)

        #Extracting the Date
        Date_sec = soup.find_all(lambda tag: tag.name == 'p' and tag.get('class') == ['_2sc7ZR'])    
        date = []
        for d in Date_sec:
          date.append(d.text)

        #Extracting the Helpful rating
        help_sec = soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['_1LmwT9'])    
        help1 = []
        for d in help_sec:
          help1.append(d.text)
    
    elif site == 'amazon':
        n_ = 0
        title_sec = soup.find_all(attrs={"data-hook": "review-title", "class": "a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold"})
        title = []
        for s in title_sec:
            title.append(s.text.replace('\n', ''))
        n_ = len(title)

        author_sec = soup.find_all(attrs = {"class": "a-profile-name"})
        author = []
        for r in author_sec:
            author.append(r.text)
        while(1):
            if len(author) > n_:
                author.pop(0)
            else:
                break

        Review_text_sec = soup.find_all(attrs={"data-hook": "review-body", "class": "a-size-base review-text review-text-content"})
        text = []
        for t in Review_text_sec:
            text.append(t.text.replace('\n', ''))

        Rating = soup.find_all(attrs={"data-hook": "review-star-rating"})    
        rate = []
        for d in Rating:
            rate.append(d.text)

        Date_sec = soup.find_all(attrs={"data-hook": "review-date"})    
        date = []
        for d in Date_sec:
          date.append(d.text) 

        help_sec = soup.find_all(attrs={"data-hook": "helpful-vote-statement"})    
        help1 = []
        for d in help_sec:
             help1.append(d.text.replace('\n          ', '')) 
        while(1):
            if len(help1) < n_:
                help1.append(0)
            else:
                break
        
    url1 = []
    url1 = [url] * len(date)

    collate = {'Date': date, 'URL': url1, 'Review_Title': title, 'Author': author, 'Rating': rate, 'Review_text': text, 'Review_helpful': help1}          
    collate_df = pd.DataFrame.from_dict(collate)
    return collate_df


def preprocess_text(text):
    stemmer = snowballstemmer.EnglishStemmer()
    text = " ".join(stemmer.stemWords(re.sub('[!"#%\'()*+,-./:;<=>?@[\\]^_`{|}~1234567890’”“′‘\\\\]', ' ', text).split(' ')))
    stop_words = set(["may", "also", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "across","among", "beside", "however", "yet", "within"] + list(ascii_lowercase))
    stop_list = stemmer.stemWords(stop_words)
    stop_words.update(stop_list)
    text = " ".join(filter(None, filter(lambda word: word not in stop_words, text.lower().split(' '))))
    return text.split(' ')

def vectorize_comments_(df, d2v_model):
    y = []
    comments = []
    for i in range(0, len(df)):
        print(i)
        label = 'SENT_%s' %i
        comments.append(d2v_model.docvecs[label])
    return comments

def scraper(url):
    df2 = []      
    soup = getsoup(url)
    site = url.split('.')[1]
    if site == 'flipkart':
        url = url + '&page=1'
    elif site == 'amazon':
        url = url + '&pageNumber=1'
    product = url.split('/')[3]
    lastPage = 1
    urllistPages = geturllist(url, lastPage)
    x = 1
    for url in urllistPages:
        soup = getsoup(url)
        df1 = getReviews(soup, site, url)         
        if x == 1:
            df3 = []
            df3 = df1
        else:                        
            df2 = df3
            result = df2.append(df1, ignore_index=True)
            df3 = result
        x += 1
    
    loaded_model = load_model('weights.best.from_scratch1 (1).hdf5')

    preprocessed_arr = [preprocess_text(x) for x in list(df3['Review_text'])]

    doc2vec_model = Doc2Vec.load("doc2vec_model_opinion_corpus (1).d2v")

    textData = vectorize_comments_(preprocessed_arr, doc2vec_model)

    textData_array = np.array(textData)

    num_vectors = textData_array.shape[0]
    textData_3d = textData_array.reshape((num_vectors, 1, -1))

    new_shape = (textData_array.shape[0], 380, 512)

    X_test3_reshaped = np.zeros(new_shape, dtype=textData_3d.dtype)
    X_test3_reshaped[:, :textData_3d.shape[1], :textData_3d.shape[2]] = textData_3d

    predictions = np.rint(loaded_model.predict(X_test3_reshaped))

    argMax = []

    for i in predictions:
        argMax.append(np.argmax(i))

    arr = []
    for i, j in enumerate(argMax):
        if j == 3 or j == 0:
            arr.append(list(df3['Review_text'])[i])
    return arr


@app.route('/', methods=['GET'])
def index():
    results = []
    if request.args.get('url'):
        results = scraper(request.args.get('url'))
    return results
    

if __name__ == "__main__":
    app.run(debug=True)
