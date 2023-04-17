from flask import Flask, url_for, redirect, render_template, request
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
model = tf.keras.models.load_model('LSTM.h5')
# nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
max_features = 20000

tokenizer = Tokenizer(
    num_words= max_features,
    lower = True,
    split=' ',
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    oov_token = '<OOK>'
)

def remove_code(text):
    
    return re.sub('<code>(.*?)</code>', '', text, flags=re.MULTILINE|re.DOTALL)
def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext
def removeLink(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)\s]','', text)
    return text
def remove_stopword(words):
    list_clean = [w for w in words.split(' ') if not w in stop_words]
    
    return ' '.join(list_clean)


def remove_next_line(words):
    words = words.split('\n')
    
    return " ".join(words)

def remove_t_char(words):
    words = words.split('\t')
    
    return "".join(words)



def remove_r_char(words):
    words = words.split('\r')
    
    return "".join(words)

lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english')) 
def text_preprocessing(text):
    text = remove_code(text)
    text =  striphtml(text)
    text = removeLink(text)
    text = clean_text(text)
    text = remove_stopword(text)
    text = remove_next_line(text)
    text = remove_t_char(text)
    text = remove_r_char(text)
                  
    text = re.sub(r'[^\w]', ' ', text)
    token_words = word_tokenize(text)
    print("token ", token_words)
    txt = [lemmatizer.lemmatize(word, pos="v") for word in token_words]
    print("txt ", txt)
    review = [word for word in txt if word not in stop_words]
    print("review ",review)
    txt = " ".join(review)
    txt = txt.strip()
    return txt

app = Flask(__name__)
app.secret_key = 'secret-key'

# def preprocess():
#     preprocessed_question = ""
#     return preprocessed_question

@app.route('/')
def index():    
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        question = request.form['que']
        q = question
        question = text_preprocessing(question)
        print("Question " , question)
        question = [question]
        tokenizer.fit_on_texts(question)
        question = tokenizer.texts_to_sequences(question)
        print(question)
        # question = [list(question)]
        print(question)
        # df = pd.DataFrame(question)
        # df = np.array(question)
        df = question
        print(df)
        df = pad_sequences(df, maxlen=300)
        print(df)
        prediction = model.predict(df)
        print("pre ",prediction)
        r = np.argmax(prediction)
        print("r ",r)
        if r == 0:
            r = "Low quality close"
        elif r == 1:
            r = "Low quality edit"
        else:
            r = "High quality"
        # if prediction == 0:
        #     prediction = "Not a duplicate question"
        # else:
        #     prediction = "Duplicate question"
        return render_template('prediction.html', prediction=r)

if __name__ == '__main__':

    app.run(port=8000, debug=True)







