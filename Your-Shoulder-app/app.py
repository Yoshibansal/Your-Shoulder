from flask import Flask, render_template
import pickle, string, util, cv2
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)


########################## ML Model ######################
# load the model
def load_model_ml():
    with open('static/models/finalized_model.pkl', 'rb') as f:
        vectorizer, clf = pickle.load(f)

    # dictionary (convert the numeric result back to text)
    dic = {4: 'sadness', 3:'anger', 1:'love', 0:'fear', 2:'joy'}

    return (vectorizer, clf, dic)


#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

# predict the output
def predict(sen, vectorizer, clf, dic):
    pun_sen = remove_punctuation(sen)
    vec_sen = vectorizer.transform([pun_sen])
    
    y_pred = clf.predict(vec_sen)
    print(f'Input sentence: {sen}')
    print("Prediction: ", dic[y_pred[0]])



######################### CNN Model ########################
def load_cnn_model():
    model = load_model('static/models/model-recent.h5')
    return model

def preprocess_image(img):
    he = util.face(img)
    img2 = he.cropDetect()

    img = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(img, (48,48), interpolation = cv2.INTER_LANCZOS4)
    print(resized.shape)

    img_batch = np.expand_dims(resized, 0)
    print(img_batch.shape)

    return img_batch

def predict_cnn(model, img_batch):
    y_val_pred = model.predict(img_batch)
    print(y_val_pred)

@app.route('/hy')
def main():
    return render_template('hybrid.html')

@app.route('/')
def hello_world():

    ## ML Model
    vectorizer, clf, dic = load_model_ml()
    sen = 'I am great..'
    predict(sen, vectorizer, clf, dic)

    img = 'static/images/j1.jpeg'
    ## CNN model
    model = load_cnn_model()
    img_batch = preprocess_image(img)
    predict_cnn(model, img_batch)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()