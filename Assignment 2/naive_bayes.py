import matplotlib.pyplot as plt 
from collections import defaultdict
import math
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from collections import defaultdict
import math
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



nltk.download('stopwords')






class NaiveBayes:
    def __init__(self):
        self.prob_classes = None
        self.word_probs = None
        self.vocab = None
        self.smoothening = None
        self.classes = None
        pass

    def get_class_prob(self,df,class_col = "label"):
        self.prob_classes ={}

        for cls in self.classes:
            class_cnt = len(df[df[class_col] == cls])
            self.prob_classes[cls] = class_cnt / len(df)

        return self.prob_classes

    def get_vocab(self,df,text_col = "content"):
        self.vocab = set()
        all_words = []
        for tokens in df[text_col]:
            all_words.extend(tokens)
            
            self.vocab.update(tokens)

        self.vocab.add("<UNK>")
        return self.vocab
    


        
    def fit(self, df, smoothening, class_col = "label", text_col = "content"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        
        self.smoothening = smoothening
        self.classes = sorted(df[class_col].unique())
        

        self.prob_classes = self.get_class_prob(df,class_col)


        
        self.vocab = self.get_vocab(df,text_col)

        self.word_probs = {}
        words_cnt = {c:defaultdict(int) for c in self.classes}   
        total_words_of_class = {c:0 for c in self.classes}

        for _, row in df.iterrows():
            cls = row[class_col]
            tokens = row[text_col]
            for token in tokens:
                words_cnt[cls][token] += 1
                total_words_of_class[cls] += 1

        vocab_size = len(self.vocab) 

        for cls in self.classes:
           
            self.word_probs[cls] = {}
            denom = total_words_of_class[cls] + smoothening * vocab_size

            for word in self.vocab:
                if word == "<UNK>":
                    cnt = 0
                else:
                    cnt = words_cnt[cls].get(word, 0)
                cnt += smoothening

                self.word_probs[cls][word] = cnt / denom


        pass
    
    def predict(self, df, text_col = "content", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
       
        predictions = []
        
      
        unknown_word_count = 0
        total_word_count = 0
        
        for _, row in df.iterrows():
            
                
            tokens = row[text_col]
            
            
            log_probs = {}
            
            for c in self.classes:
               
                log_probs[c] = math.log(self.prob_classes[c])
                
                
                for token in tokens:
                    total_word_count += 1
                    
                    if token not in self.word_probs[c]:
                        token = "<UNK>"
                        unknown_word_count += 1
                    
                    log_probs[c] += math.log(self.word_probs[c][token])
            
           
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)
        df[predicted_col] = predictions    
        print(self.accuracy(df, class_col = "label", predicted_col = predicted_col))
        pass

    def accuracy(self, df, class_col = "label", predicted_col = "Predicted"):
        """
        Compute the accuracy of the model on the given data.

        Args:
            df (pd.DataFrame): The data containing columns class_col and predicted_col.
        Returns:
            float: The accuracy of the model on the given data.
        """
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total
    
    def plot_wordcloud(self, class_label, max_words=100):

        frequency_of_words = self.word_probs[class_label]

        wc = WordCloud(width=800, height=400, 
                       background_color='white', 
                       max_words=max_words).generate_from_frequencies(frequency_of_words)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for class {class_label}", fontsize=16)
        plt.show()



class NaiveBayesPreprocess:
    def __init__(self):
        self.prob_classes = None
        self.word_probs = None
        self.vocab = None
        self.smoothening = None
        self.classes = None
        pass
    
    def get_class_prob(self,df,class_col = "label"):
        self.prob_classes ={}

        for cls in self.classes:
            class_cnt = len(df[df[class_col] == cls])
            self.prob_classes[cls] = class_cnt / len(df)

        return self.prob_classes

    def get_vocab(self,df,text_col = "content"):
        self.vocab = set()
        all_words = []
        for tokens in df[text_col]:
            all_words.extend(tokens)
            
            self.vocab.update(tokens)

        self.vocab.add("<UNK>")
        return self.vocab


    def preprocess_text(self,df, text_col="content"):

        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        
        df[text_col] = df[text_col].apply(
            lambda tokens: [
                stemmer.stem(w) 
                for w in tokens 
                if w.lower() not in stop_words and w.isalpha() 
            ]
        )
        return df

        
    def fit(self, df, smoothening, class_col = "label", text_col = "content"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        df = self.preprocess_text(df, text_col)
        self.smoothening = smoothening
        self.classes = sorted(df[class_col].unique())
        

        self.prob_classes = self.get_class_prob(df,class_col)


        
        self.vocab = self.get_vocab(df,text_col)
        self.word_probs = {}
        words_cnt = {c:defaultdict(int) for c in self.classes}   
        total_words_of_class = {c:0 for c in self.classes}

        for _, row in df.iterrows():
            cls = row[class_col]
            tokens = row[text_col]
            for token in tokens:
                words_cnt[cls][token] += 1
                total_words_of_class[cls] += 1

        vocab_size = len(self.vocab) 

        for cls in self.classes:
            
            self.word_probs[cls] = {}
            denom = total_words_of_class[cls] + smoothening * vocab_size

            for word in self.vocab:
                if word == "<UNK>":
                    cnt = 0
                else:
                    cnt = words_cnt[cls].get(word, 0)
                cnt += smoothening

                self.word_probs[cls][word] = cnt / denom


        pass
    
    def predict(self, df, text_col = "content", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        df = self.preprocess_text(df, text_col)
        predictions = []
        
      
        unknown_word_count = 0
        total_word_count = 0
        
        for _, row in df.iterrows():
            
                
            tokens = row[text_col]
            
            
            log_probs = {}
            
            for c in self.classes:
               
                log_probs[c] = math.log(self.prob_classes[c])
                
                
                for token in tokens:
                    total_word_count += 1
                    
                    if token not in self.word_probs[c]:
                        token = "<UNK>"
                        unknown_word_count += 1
                    
                    log_probs[c] += math.log(self.word_probs[c][token])
            
           
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)
        df[predicted_col] = predictions    
        print(self.accuracy(df, class_col = "label", predicted_col = predicted_col))
        pass

    def accuracy(self, df, class_col = "label", predicted_col = "Predicted"):
        """
        Compute the accuracy of the model on the given data.

        Args:
            df (pd.DataFrame): The data containing columns class_col and predicted_col.
        Returns:
            float: The accuracy of the model on the given data.
        """
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total
    
    def plot_wordcloud(self, class_label, max_words=100):
        """
        Generate and display a word cloud for the given class.
        Args:
            class_label: the class to visualize
            max_words: max number of words in the word cloud
        """
        if class_label not in self.word_probs:
            raise ValueError(f"Class {class_label} not found. Available: {list(self.word_probs.keys())}")
        
        word_freqs = self.word_probs[class_label]

        wc = WordCloud(width=800, height=400, 
                       background_color='white', 
                       max_words=max_words).generate_from_frequencies(word_freqs)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for class {class_label}", fontsize=16)
        plt.show()
        

class NaiveBayesBigram:
    def __init__(self):
        self.prob_classes = None
        self.word_probs = None
        self.vocab = None
        self.smoothening = None
        self.classes = None
        pass

    def make_bigrams(self,tokens):
        bigrams = []
        bigrams.extend(tokens)
        for i in range(len(tokens)-1):
            bigrams.append(tokens[i] + " " + tokens[i+1])
        return bigrams    
    

    def preprocess_text(self,df, text_col="content"):
        
        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        
        df[text_col] = df[text_col].apply(
            lambda tokens: [
                stemmer.stem(w) 
                for w in tokens 
                if w.lower() not in stop_words and w.isalpha()  # keep only non-stopwords + alphabetic
            ]
        )
        return df
    
    def get_class_prob(self,df,class_col = "label"):
        self.prob_classes ={}

        for cls in self.classes:
            class_cnt = len(df[df[class_col] == cls])
            self.prob_classes[cls] = class_cnt / len(df)

        return self.prob_classes

    def get_vocab(self,df,text_col = "content"):
        self.vocab = set()
        all_words = []
        for tokens in df[text_col]:
            all_words.extend(tokens)
            
            self.vocab.update(tokens)

        self.vocab.add("<UNK>")
        return self.vocab

        
    def fit(self, df, smoothening, class_col = "label", text_col = "content"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        df = self.preprocess_text(df, text_col)
        self.smoothening = smoothening
        self.classes = sorted(df[class_col].unique())
        

        self.prob_classes = self.get_class_prob(df,class_col)


        
        self.vocab = self.get_vocab(df,text_col)

        self.word_probs = {}
        words_cnt = {c:defaultdict(int) for c in self.classes}   
        total_words_of_class = {c:0 for c in self.classes}

        for _, row in df.iterrows():
            cls = row[class_col]
            tokens = row[text_col]
            features = self.make_bigrams(tokens)
            for feature in features:
                words_cnt[cls][feature] += 1
                total_words_of_class[cls] += 1

        vocab_size = len(self.vocab) 

        for cls in self.classes:
           
            self.word_probs[cls] = {}
            denom = total_words_of_class[cls] + smoothening * vocab_size

            for word in self.vocab:
                if word == "<UNK>":
                    cnt = 0
                else:
                    cnt = words_cnt[cls].get(word, 0)
                cnt += smoothening

                self.word_probs[cls][word] = cnt / denom


        pass
    
    def predict(self, df, text_col = "content", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        df = self.preprocess_text(df, text_col)
        predictions = []
        
      
        unknown_word_count = 0
        total_word_count = 0
        
        for _, row in df.iterrows():
            
                
            tokens = row[text_col]
            features = self.make_bigrams(tokens)
            
            
            log_probs = {}
            
            for c in self.classes:
               
                log_probs[c] = math.log(self.prob_classes[c])
                
                
                for token in features:
                    total_word_count += 1
                    
                    if token not in self.word_probs[c]:
                        token = "<UNK>"
                        unknown_word_count += 1
                    
                    log_probs[c] += math.log(self.word_probs[c][token])
            
           
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)
        df[predicted_col] = predictions    
        
        print(self.accuracy(df, class_col = "label", predicted_col = predicted_col))
        print(f"unknown_word_count: {unknown_word_count}, total_word_count: {total_word_count}")
        pass

    def accuracy(self, df, class_col = "label", predicted_col = "Predicted"):
        
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total
    
    def plot_wordcloud(self, class_label, max_words=100):
        
        if class_label not in self.word_probs:
            raise ValueError(f"Class {class_label} not found. Available: {list(self.word_probs.keys())}")
        
        word_freqs = self.word_probs[class_label]

        wc = WordCloud(width=800, height=400, 
                       background_color='white', 
                       max_words=max_words).generate_from_frequencies(word_freqs)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for class {class_label}", fontsize=16)
        plt.show()
        


class NaiveBayesWithoutPreprocessBigram:
    def __init__(self):
        self.prob_classes = None
        self.word_probs = None
        self.vocab = None
        self.smoothening = None
        self.classes = None
        pass

    def make_bigrams(self,tokens):
        bigrams = []
        bigrams.extend(tokens)
        for i in range(len(tokens)-1):
            bigrams.append(tokens[i] + " " + tokens[i+1])
        return bigrams    
    
    def get_class_prob(self,df,class_col = "label"):
        self.prob_classes ={}

        for cls in self.classes:
            class_cnt = len(df[df[class_col] == cls])
            self.prob_classes[cls] = class_cnt / len(df)

        return self.prob_classes

    def get_vocab(self,df,text_col = "content"):
        self.vocab = set()
        all_words = []
        for tokens in df[text_col]:
            all_words.extend(tokens)
            
            self.vocab.update(tokens)

        self.vocab.add("<UNK>")
        return self.vocab
    

    def preprocess_text(self,df, text_col="content"):
        """
        Apply stemming + stopword removal to tokenized text in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with tokenized text column
            text_col (str): Column containing list of tokens
        Returns:
            pd.DataFrame with updated text_col
        """
        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        
        df[text_col] = df[text_col].apply(
            lambda tokens: [
                stemmer.stem(w) 
                for w in tokens 
                if w.lower() not in stop_words and w.isalpha()  # keep only non-stopwords + alphabetic
            ]
        )
        return df

        
    def fit(self, df, smoothening, class_col = "label", text_col = "content"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        # df = self.preprocess_text(df, text_col)
        self.smoothening = smoothening
        self.classes = sorted(df[class_col].unique())
        

        self.prob_classes = self.get_class_prob(df,class_col)


        
        self.vocab = self.get_vocab(df,text_col)
        self.word_probs = {}

        words_cnt = {c:defaultdict(int) for c in self.classes}   
        total_words_of_class = {c:0 for c in self.classes}

        for _, row in df.iterrows():
            cls = row[class_col]
            tokens = row[text_col]
            features = self.make_bigrams(tokens)
            for feature in features:
                words_cnt[cls][feature] += 1
                total_words_of_class[cls] += 1

        vocab_size = len(self.vocab) 

        for cls in self.classes:
           
            self.word_probs[cls] = {}
            denom = total_words_of_class[cls] + smoothening * vocab_size

            for word in self.vocab:
                if word == "<UNK>":
                    cnt = 0
                else:
                    cnt = words_cnt[cls].get(word, 0)
                cnt += smoothening

                self.word_probs[cls][word] = cnt / denom


        pass
    
    def predict(self, df, text_col = "content", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        # df = self.preprocess_text(df, text_col)
        predictions = []
        
      
        unknown_word_count = 0
        total_word_count = 0
        
        for _, row in df.iterrows():
            
                
            tokens = row[text_col]
            features = self.make_bigrams(tokens)
            
            
            log_probs = {}
            
            for c in self.classes:
               
                log_probs[c] = math.log(self.prob_classes[c])
                
                
                for token in features:
                    total_word_count += 1
                    
                    if token not in self.word_probs[c]:
                        token = "<UNK>"
                        unknown_word_count += 1
                    
                    log_probs[c] += math.log(self.word_probs[c][token])
            
           
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)
        df[predicted_col] = predictions    
        
        print(self.accuracy(df, class_col = "label", predicted_col = predicted_col))
        print(f"unknown_word_count: {unknown_word_count}, total_word_count: {total_word_count}")
        pass

    def accuracy(self, df, class_col = "label", predicted_col = "Predicted"):
        """
        Compute the accuracy of the model on the given data.

        Args:
            df (pd.DataFrame): The data containing columns class_col and predicted_col.
        Returns:
            float: The accuracy of the model on the given data.
        """
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total
    
    def plot_wordcloud(self, class_label, max_words=100):
        """
        Generate and display a word cloud for the given class.
        Args:
            class_label: the class to visualize
            max_words: max number of words in the word cloud
        """
        if class_label not in self.word_probs:
            raise ValueError(f"Class {class_label} not found. Available: {list(self.word_probs.keys())}")
        
        word_freqs = self.word_probs[class_label]

        wc = WordCloud(width=800, height=400, 
                       background_color='white', 
                       max_words=max_words).generate_from_frequencies(word_freqs)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for class {class_label}", fontsize=16)
        plt.show()




class NaiveBayesBigramTemp:
    def __init__(self):
        self.prob_classes = None
        self.word_probs = None
        self.vocab = None
        self.smoothening = None
        self.classes = None
        pass

    def make_bigrams(self,tokens):
        bigrams = []
        # bigrams.extend(tokens)
        for i in range(len(tokens)-1):
            bigrams.append(tokens[i] + " " + tokens[i+1])
        return bigrams    
    

    def preprocess_text(self,df, text_col="content"):
        """
        Apply stemming + stopword removal to tokenized text in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with tokenized text column
            text_col (str): Column containing list of tokens
        Returns:
            pd.DataFrame with updated text_col
        """
        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        
        df[text_col] = df[text_col].apply(
            lambda tokens: [
                stemmer.stem(w) 
                for w in tokens 
                if w.lower() not in stop_words and w.isalpha()  # keep only non-stopwords + alphabetic
            ]
        )
        return df
    def get_class_prob(self,df,class_col = "label"):
        self.prob_classes ={}

        for cls in self.classes:
            class_cnt = len(df[df[class_col] == cls])
            self.prob_classes[cls] = class_cnt / len(df)

        return self.prob_classes

    def get_vocab(self,df,text_col = "content"):
        self.vocab = set()
        all_words = []
        for tokens in df[text_col]:
            all_words.extend(tokens)
            
            self.vocab.update(tokens)

        self.vocab.add("<UNK>")
        return self.vocab

        
    def fit(self, df, smoothening, class_col = "label", text_col = "content"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        df = self.preprocess_text(df, text_col)
        self.smoothening = smoothening
        self.classes = sorted(df[class_col].unique())
        

        self.prob_classes = self.get_class_prob(df,class_col)


        
        self.vocab = self.get_vocab(df,text_col)
        self.word_probs = {}
        words_cnt = {c:defaultdict(int) for c in self.classes}   
        total_words_of_class = {c:0 for c in self.classes}

        for _, row in df.iterrows():
            cls = row[class_col]
            tokens = row[text_col]
            features = self.make_bigrams(tokens)
            for feature in features:
                words_cnt[cls][feature] += 1
                total_words_of_class[cls] += 1

        vocab_size = len(self.vocab) 

        for cls in self.classes:
           
            self.word_probs[cls] = {}
            denom = total_words_of_class[cls] + smoothening * vocab_size

            for word in self.vocab:
                if word == "<UNK>":
                    cnt = 0
                else:
                    cnt = words_cnt[cls].get(word, 0)
                cnt += smoothening

                self.word_probs[cls][word] = cnt / denom


        pass
    
    def predict(self, df, text_col = "content", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        df = self.preprocess_text(df, text_col)
        predictions = []
        
      
        unknown_word_count = 0
        total_word_count = 0
        
        for _, row in df.iterrows():
            
                
            tokens = row[text_col]
            features = self.make_bigrams(tokens)
            
            
            log_probs = {}
            
            for c in self.classes:
               
                log_probs[c] = math.log(self.prob_classes[c])
                
                
                for token in features:
                    total_word_count += 1
                    
                    if token not in self.word_probs[c]:
                        token = "<UNK>"
                        unknown_word_count += 1
                    
                    log_probs[c] += math.log(self.word_probs[c][token])
            
           
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)
        df[predicted_col] = predictions    
        
        print(self.accuracy(df, class_col = "label", predicted_col = predicted_col))
        print(f"unknown_word_count: {unknown_word_count}, total_word_count: {total_word_count}")
        pass

    def accuracy(self, df, class_col = "label", predicted_col = "Predicted"):
        """
        Compute the accuracy of the model on the given data.

        Args:
            df (pd.DataFrame): The data containing columns class_col and predicted_col.
        Returns:
            float: The accuracy of the model on the given data.
        """
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total
    
    def plot_wordcloud(self, class_label, max_words=100):
        """
        Generate and display a word cloud for the given class.
        Args:
            class_label: the class to visualize
            max_words: max number of words in the word cloud
        """
        if class_label not in self.word_probs:
            raise ValueError(f"Class {class_label} not found. Available: {list(self.word_probs.keys())}")
        
        word_freqs = self.word_probs[class_label]

        wc = WordCloud(width=800, height=400, 
                       background_color='white', 
                       max_words=max_words).generate_from_frequencies(word_freqs)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for class {class_label}", fontsize=16)
        plt.show()



class NaiveBayesMultifeature:
    def __init__(self):
        self.prob_classes = {}
        self.word_probs = {}
        self.vocab = {}
        self.smoothening = None
        self.classes = None
        self.feature_types = ['content', 'title']
        pass

    def make_bigrams(self,tokens):
        bigrams = []
        bigrams.extend(tokens)
        for i in range(len(tokens)-1):
            bigrams.append(tokens[i] + " " + tokens[i+1])
        return bigrams    
    

    def preprocess_text(self,df, text_col="content"):
        """
        Apply stemming + stopword removal to tokenized text in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with tokenized text column
            text_col (str): Column containing list of tokens
        Returns:
            pd.DataFrame with updated text_col
        """
        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        
        df[text_col] = df[text_col].apply(
            lambda tokens: [
                stemmer.stem(w) 
                for w in tokens 
                if w.lower() not in stop_words and w.isalpha()  # keep only non-stopwords + alphabetic
            ]
        )
        return df
    
    def get_class_probabilities(self,df,class_col = "label"):
        prob_classes = {}
        classes = sorted(df[class_col].unique())
        for cls in classes:
            class_cnt = len(df[df[class_col] == cls])
            prob_classes[cls] = class_cnt / len(df)
        return prob_classes, classes

        
    def fit(self, df, smoothening,class_col = "label"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        for feature_type in self.feature_types:
            df = self.preprocess_text(df, feature_type)

        
        self.smoothening = smoothening

        
        self.prob_classes, self.classes = self.get_class_probabilities(df, class_col)
        
      

        for feature_type in self.feature_types:
            self.vocab[feature_type] = set()
            self.word_probs[feature_type] = {} 

            all_words = []
            for tokens in df[feature_type]:
                features = self.make_bigrams(tokens)
                all_words.extend(features)
                self.vocab[feature_type].update(features)

            self.vocab[feature_type].add("<UNK>")



        token_count = {feature_type : { c : defaultdict(int) for c in self.classes} for feature_type in self.feature_types}
        total_token_per_class = {feature_type : { c : 0 for c in self.classes} for feature_type in self.feature_types}



        for feature_type in self.feature_types:
            for _,row in df.iterrows():
                cls = row[class_col]
                tokens = row[feature_type]
                features = self.make_bigrams(tokens)

                for feature in features :

                    token_count[feature_type][cls][feature] += 1
                    total_token_per_class[feature_type][cls] += 1

        for feature_type in self.feature_types:
            vocab_size = len(self.vocab[feature_type]) 

            for cls in self.classes:
                self.word_probs[feature_type][cls] = {}

                denom = total_token_per_class[feature_type][cls] + smoothening * vocab_size

                for word in self.vocab[feature_type]:
                    if word == "<UNK>":
                        cnt = 0
                    else:
                        cnt = token_count[feature_type][cls].get(word, 0)
                    cnt += smoothening
                    self.word_probs[feature_type][cls][word] = cnt / denom        



        pass
    
    def predict(self, df, predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.

                
        """
        for feature_type in self.feature_types:
            df = self.preprocess_text(df, feature_type)


        predictions = []
        unknown_word_count = 0
        total_word_count = 0


        for _ , row in df.iterrows():
            log_probs = {}

            for c in self.classes:
                log_probs[c] = math.log(self.prob_classes[c])

                for feature_type in self.feature_types:
                    tokens = row[feature_type]

                    features = self.make_bigrams(tokens)

                    for feature in features:
                        if feature not in self.word_probs[feature_type][c]:
                            feature = "<UNK>"
                            unknown_word_count += 1
                        log_probs[c] += math.log(self.word_probs[feature_type][c][feature])
                        total_word_count += 1
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)

        df[predicted_col] = predictions

          
        
        print(self.accuracy(df, class_col = "label", predicted_col = predicted_col))
        print(f"unknown_word_count: {unknown_word_count}, total_word_count: {total_word_count}")
        pass

    def accuracy(self, df, class_col = "label", predicted_col = "Predicted"):
        """
        Compute the accuracy of the model on the given data.

        Args:
            df (pd.DataFrame): The data containing columns class_col and predicted_col.
        Returns:
            float: The accuracy of the model on the given data.
        """
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total
    
    def plot_wordcloud(self, class_label, max_words=100):
        """
        Generate and display a word cloud for the given class.
        Args:
            class_label: the class to visualize
            max_words: max number of words in the word cloud
        """
        if class_label not in self.word_probs:
            raise ValueError(f"Class {class_label} not found. Available: {list(self.word_probs.keys())}")
        
        word_freqs = self.word_probs[class_label]

        wc = WordCloud(width=800, height=400, 
                       background_color='white', 
                       max_words=max_words).generate_from_frequencies(word_freqs)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for class {class_label}", fontsize=16)
        plt.show()
        


class NaiveBayesMultifeatureTrigram:

    def __init__(self):
        self.prob_classes = {}
        self.word_probs = {}
        self.vocab = {}
        self.smoothening = None
        self.classes = None
        self.feature_types = ['content', 'title']
        pass

    def make_bigrams(self,tokens):
        trigrams = []
        trigrams.extend(tokens)
        for i in range(len(tokens)-2):
            trigrams.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2])
        return trigrams
    
    def get_class_probabilities(self,df,class_col = "label"):
        prob_classes = {}
        classes = sorted(df[class_col].unique())
        for cls in classes:
            class_cnt = len(df[df[class_col] == cls])
            prob_classes[cls] = class_cnt / len(df)
        return prob_classes, classes

    def preprocess_text(self,df, text_col="content"):
        """
        Apply stemming + stopword removal to tokenized text in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with tokenized text column
            text_col (str): Column containing list of tokens
        Returns:
            pd.DataFrame with updated text_col
        """
        stemmer = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        
        df[text_col] = df[text_col].apply(
            lambda tokens: [
                stemmer.stem(w) 
                for w in tokens 
                if w.lower() not in stop_words and w.isalpha()  # keep only non-stopwords + alphabetic
            ]
        )
        return df

        
    def fit(self, df, smoothening,class_col = "label"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        for feature_type in self.feature_types:
            df = self.preprocess_text(df, feature_type)

        
        self.smoothening = smoothening
        self.classes = sorted(df[class_col].unique())
        

        self.prob_classes, self.classes = self.get_class_probabilities(df, class_col)

        for feature_type in self.feature_types:
            self.vocab[feature_type] = set()
            self.word_probs[feature_type] = {} 

            all_words = []
            for tokens in df[feature_type]:
                features = self.make_bigrams(tokens)
                all_words.extend(features)
                self.vocab[feature_type].update(features)

            self.vocab[feature_type].add("<UNK>")



        token_count = {feature_type : { c : defaultdict(int) for c in self.classes} for feature_type in self.feature_types}
        total_token_per_class = {feature_type : { c : 0 for c in self.classes} for feature_type in self.feature_types}



        for feature_type in self.feature_types:
            for _,row in df.iterrows():
                cls = row[class_col]
                tokens = row[feature_type]
                features = self.make_bigrams(tokens)

                for feature in features :

                    token_count[feature_type][cls][feature] += 1
                    total_token_per_class[feature_type][cls] += 1

        for feature_type in self.feature_types:
            vocab_size = len(self.vocab[feature_type]) 

            for cls in self.classes:
                self.word_probs[feature_type][cls] = {}

                denom = total_token_per_class[feature_type][cls] + smoothening * vocab_size

                for word in self.vocab[feature_type]:
                    if word == "<UNK>":
                        cnt = 0
                    else:
                        cnt = token_count[feature_type][cls].get(word, 0)
                    cnt += smoothening
                    self.word_probs[feature_type][cls][word] = cnt / denom        



        pass
    
    def predict(self, df, predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.

                
        """
        for feature_type in self.feature_types:
            df = self.preprocess_text(df, feature_type)


        predictions = []
        unknown_word_count = 0
        total_word_count = 0


        for _ , row in df.iterrows():
            log_probs = {}

            for c in self.classes:
                log_probs[c] = math.log(self.prob_classes[c])

                for feature_type in self.feature_types:
                    tokens = row[feature_type]

                    features = self.make_bigrams(tokens)

                    for feature in features:
                        if feature not in self.word_probs[feature_type][c]:
                            feature = "<UNK>"
                            unknown_word_count += 1
                        log_probs[c] += math.log(self.word_probs[feature_type][c][feature])
                        total_word_count += 1
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)

        df[predicted_col] = predictions

          
        
        print(self.accuracy(df, class_col = "label", predicted_col = predicted_col))
        print(f"unknown_word_count: {unknown_word_count}, total_word_count: {total_word_count}")
        pass

    def accuracy(self, df, class_col = "label", predicted_col = "Predicted"):
        """
        Compute the accuracy of the model on the given data.

        Args:
            df (pd.DataFrame): The data containing columns class_col and predicted_col.
        Returns:
            float: The accuracy of the model on the given data.
        """
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total
    
    def plot_wordcloud(self, class_label, max_words=100):
        """
        Generate and display a word cloud for the given class.
        Args:
            class_label: the class to visualize
            max_words: max number of words in the word cloud
        """
        if class_label not in self.word_probs:
            raise ValueError(f"Class {class_label} not found. Available: {list(self.word_probs.keys())}")
        
        word_freqs = self.word_probs[class_label]

        wc = WordCloud(width=800, height=400, 
                       background_color='white', 
                       max_words=max_words).generate_from_frequencies(word_freqs)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for class {class_label}", fontsize=16)
        plt.show()





def analysis():
    
    data = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    data['content'] = data['content'].str.split()
    data['title'] = data['title'].str.split()
    data_test['title'] = data_test['title'].str.split()
    data_test['content'] = data_test['content'].str.split()




    def accuracy(df, class_col = "label", predicted_col = "Predicted"):
        correct = (df[class_col] == df[predicted_col]).sum()
        total = len(df)
        return correct / total

    def precision_recall(df, class_col = "label", predicted_col = "Predicted", positive_class=1):
        tp = ((df[class_col] == positive_class) & (df[predicted_col] == positive_class)).sum()
        fp = ((df[class_col] != positive_class) & (df[predicted_col] == positive_class)).sum()
        fn = ((df[class_col] == positive_class) & (df[predicted_col] != positive_class)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recall

    def f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0        




    NB = NaiveBayes()
    NB.fit(data, smoothening = 1.0, class_col = "label", text_col = "content")
    NB.predict(data_test, text_col = "content", predicted_col = "Predicted")
    NB.predict(data, text_col = "content", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")

    for i in range(14):
        NB.plot_wordcloud(i, max_words=100)

    NB1 = NaiveBayesPreprocess()
    NB1.fit(data, smoothening = 1.0, class_col = "label", text_col = "content")
    NB1.predict(data_test, text_col = "content", predicted_col = "Predicted")
    NB1.predict(data, text_col = "content", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")


    for i in range(14):
        NB1.plot_wordcloud(i, max_words=100)

    NB2 = NaiveBayesBigram()
    NB2.fit(data, smoothening = 1.0, class_col = "label", text_col = "content")
    NB2.predict(data_test, text_col = "content", predicted_col = "Predicted")
    NB2.predict(data, text_col = "content", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")

    data = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    data['content'] = data['content'].str.split()
    data['title'] = data['title'].str.split()
    data_test['title'] = data_test['title'].str.split()
    data_test['content'] = data_test['content'].str.split()

    NB3 = NaiveBayesWithoutPreprocessBigram()
    NB3.fit(data, smoothening = 1.0, class_col = "label", text_col = "content")
    NB3.predict(data_test, text_col = "content", predicted_col = "Predicted")
    NB3.predict(data, text_col = "content", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")


    NBtemp = NaiveBayesBigramTemp()
    NBtemp.fit(data, smoothening = 1.0, class_col = "label", text_col = "content")
    NBtemp.predict(data_test, text_col = "content", predicted_col = "Predicted")
    NBtemp.predict(data, text_col = "content", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")




    NB = NaiveBayes()
    NB.fit(data, smoothening = 1.0, class_col = "label", text_col = "title")
    NB.predict(data_test, text_col = "title", predicted_col = "Predicted")
    NB.predict(data, text_col = "title", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")
        

    NB1 = NaiveBayesPreprocess()
    NB1.fit(data, smoothening = 1.0, class_col = "label", text_col = "title")
    NB1.predict(data_test, text_col = "title", predicted_col = "Predicted")
    NB1.predict(data, text_col = "title", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")


    NB2 = NaiveBayesBigram()
    NB2.fit(data, smoothening = 1.0, class_col = "label", text_col = "title")
    NB2.predict(data_test, text_col = "title", predicted_col = "Predicted")
    NB2.predict(data, text_col = "title", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")


    NB2 = NaiveBayesWithoutPreprocessBigram()
    NB2.fit(data, smoothening = 1.0, class_col = "label", text_col = "title")
    NB2.predict(data_test, text_col = "title", predicted_col = "Predicted")
    NB2.predict(data, text_col = "title", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")



    data['content+title'] =  data['title'] + data['content'] 
    data_test['content+title'] =  data_test['title'] + data_test['content']
    NB2 = NaiveBayes()
    NB2.fit(data, smoothening = 1.0, class_col = "label", text_col = "content+title")
    NB2.predict(data_test, text_col = "content+title", predicted_col = "Predicted")
    NB2.predict(data, text_col = "content+title", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")

    NB2 = NaiveBayesBigram()
    NB2.fit(data, smoothening = 1.0, class_col = "label", text_col = "content+title")
    NB2.predict(data_test, text_col = "content+title", predicted_col = "Predicted")
    NB2.predict(data, text_col = "content+title", predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}")



    data = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    data['content'] = data['content'].str.split()
    data['title'] = data['title'].str.split()
    data_test['title'] = data_test['title'].str.split()
    data_test['content'] = data_test['content'].str.split()


    NB_Multi = NaiveBayesMultifeature()
    NB_Multi.fit(data, smoothening = 1.0, class_col = "label")
    NB_Multi.predict(data_test, predicted_col = "Predicted")
    NB_Multi.predict(data, predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}") 

    data_test['random_predict'] = [random.randint(0,13) for _ in range(len(data_test))]
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'random_predict')}")


    def plot_confusion_matrix(df, true_col="label", pred_col="Predicted"):
        # Compute confusion matrix
        cm = confusion_matrix(df[true_col], df[pred_col])
        labels = sorted(df[true_col].unique())

        # Create a bigger, clearer plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True,          # display numbers
            fmt='d',             # integer format
            cmap='Blues',        # color map
            xticklabels=labels, 
            yticklabels=labels,
            cbar=True,
            annot_kws={"size": 9}  # adjust font size for annotations
        )

        plt.title("Confusion Matrix", fontsize=16, pad=15)
        plt.xlabel("Predicted Class", fontsize=14)
        plt.ylabel("True Class", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.show()
    plot_confusion_matrix(data_test, true_col="label", pred_col="random_predict")

    random_predict = random.randint(0,13)
    data_test['random_predict'] = [random_predict for _ in range(len(data_test))]
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'random_predict')}")
    plot_confusion_matrix(data_test, true_col="label", pred_col="random_predict")

    plot_confusion_matrix(data_test, true_col="label", pred_col="Predicted")





    data = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    data['content'] = data['content'].str.split()
    data['title'] = data['title'].str.split()
    data_test['title'] = data_test['title'].str.split()
    data_test['content'] = data_test['content'].str.split()

    NB_Multi = NaiveBayesMultifeatureTrigram()
    NB_Multi.fit(data, smoothening = 1.0, class_col = "label")
    NB_Multi.predict(data_test, predicted_col = "Predicted")
    NB_Multi.predict(data, predicted_col = "Predicted")
    print(f"accuracy on test data: {accuracy(data_test, class_col = 'label', predicted_col = 'Predicted')}")
    for i in range(14):
        print(f"precison and recall for class {i}: {precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i)}")
        print(f"f1 score for class {i}: {f1_score(*precision_recall(data_test, class_col = 'label', predicted_col = 'Predicted', positive_class=i))}") 



if __name__ == "__main__":
    data = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    data['content'] = data['content'].str.split()
    data['title'] = data['title'].str.split()
    data_test['title'] = data_test['title'].str.split()
    data_test['content'] = data_test['content'].str.split()

    analysis()



















