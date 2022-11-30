import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import nltk
import spacy
import networkx as nx


class NBClassifier:
    def __init__(self, use_sdp=False):
        self.pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=self._vectorizer_tokenize, binary=True, lowercase=False)),
            ('clf', MultiOutputClassifier(BernoulliNB()))
        ])
        self._nlp = spacy.load('en_core_web_sm')
        self._use_sdp = use_sdp

    def train(self, X, y):
        self.pipeline.fit([self._preprocess(x) for x in X], y)

    def predict_label(self, doc, ent1, ent2):
        return self.pipeline.predict([self._preprocess(doc, ent1, ent2)])
    
    def _preprocess(self, doc, ent1='food_entity', ent2='disease_entity'):
        doc = doc.replace(ent1, 'food_entity')
        doc = doc.replace(ent2, 'disease_entity')
        if self._use_sdp:
            tokens = doc.lower()
            tokens = self._shortest_dep_path(doc, ent1, ent2)
            return [x for x in tokens if x not in nltk.corpus.stopwords.words('english') and len(x) > 1]
        else:
            tokens = doc.lower()
            tokens = nltk.word_tokenize(tokens)
            tokens = [x for x in tokens if x not in nltk.corpus.stopwords.words('english') and len(x) > 1]
            lemmatizer = nltk.stem.WordNetLemmatizer()
            return [lemmatizer.lemmatize(token) for token in tokens]

    def _shortest_dep_path(self, doc, entity1, entity2):
        doc = self._nlp(doc)
        edges = []
        for token in doc:
            for child in token.children:
                edges.append((
                    '{0}'.format(token.lemma_),
                    '{0}'.format(child.lemma_)))
        graph = nx.Graph(edges)
        try:
            return nx.shortest_path(graph, source=entity1, target=entity2)
        except:
            return []
    
    def _vectorizer_tokenize(self, x):
        return x
    
    def save_model(self, filename):
        joblib.dump(self.pipeline, filename)

    def load_model(self, filename):
        self.pipeline = joblib.load(filename)
