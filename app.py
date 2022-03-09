from flask import Flask, render_template, request
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()


classifier = pickle.load(open('model1.pkl', 'rb'))
cv = pickle.load(open('cv1.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if (request.method == 'POST'):
        
        review = request.form['message']
        stop_words = stopwords.words('english')
        stop_words.remove('not')
        data = []
        review = review.lower()
        review = review.split()
        review = [porter.stem(word) for word in review if not word in stop_words]
       
        review = ' '.join(review)
        data.append(review)

        vect = cv.transform(data).toarray()
        prediction = classifier.predict(vect)

        if (prediction == 1 or prediction == 2):
            output = "Negative Review"
        elif (prediction == 3):
            output = "Neutral  Review"
        elif (prediction == 4 or prediction == 5):
            output = "Positive Review"        


        return render_template('index.html',prediction_text= output)
    else:
        return render_template('index.html')    

if __name__ == '__main__':
    app.run(debug=True)
