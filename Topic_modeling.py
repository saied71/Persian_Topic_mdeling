import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

class Topic_modeling:
    
    def __init__(self, *args, **kwargs):
            pass
    
    def read(self, path):
        data = pd.read_pickle(path)
        df = pd.DataFrame({"col":data})
        return df
    
    
    def tokenize(self, string):
        tokens = word_tokenize(string)
        tokens = " ".join(tokens)
        tokens = re.sub("[\s]", " ", tokens)
        tokens = re.sub('[/(){}\[\]\|@,;!٪×،*ـ+؟؛"" ... .. . <> _ - :]', " ", tokens)
        tokens = re.sub('[!٬٫﷼٪×*)(ـ+}|؛؟<>‌ ÷؛«» "" -]', " ", tokens)
        clean_tokens = [w for w in tokens.split(" ") if not w in stopwords.words("persian")]
        final_token = [w for w in clean_tokens if len(w)>2]
        return final_token
    
    
    def tfidf_vec(self, DF):
        tfidf = TfidfVectorizer(max_df=.95, min_df=2, tokenizer=self.tokenize)
        per_tfidf = tfidf.fit_transform(DF.col)
        return per_tfidf, tfidf
    
    
    def NMF(self, path, n_topic):
        nmf_model = NMF(n_components=n_topic, random_state=42)
        df = self.read(path)
        per_tfidf, tfidf = self.tfidf_vec(df)
        nmf_model.fit(per_tfidf)
        l = []
        for index,topic in enumerate(nmf_model.components_):
#             print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
            l.append([tfidf.get_feature_names()[i] for i in topic.argsort()[-18:]])
        DF_words = pd.DataFrame({"Top words for topic":l})
        topic_results = nmf_model.transform(per_tfidf)
        df['topic'] = topic_results.argmax(axis=1)
        return df, DF_words