import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        return

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        models = []
        num_features = self.X.shape[1]
        num_params = []
        for n in range(self.min_n_components, self.max_n_components+1):
            models.append(self.base_model(n))
            num_param = n * (num_features * 2 + 1)
            num_params.append(num_param)

        best_score = float("inf")
        best_model = models[0]
        for model, p in zip(models, num_params):
            try:
                model.fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                N = len(self.X)
                score = -2 * logL + p * np.log(N) 
            except:
                score = float("inf")
            if score < best_score:
                best_score = score
                best_model = model
        return best_model



class SelectorDIC(ModelSelector):
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        models = []
        for n in range(self.min_n_components, self.max_n_components+1):
            models.append(self.base_model(n))
            
        best_score = float("-inf")
        best_model = models[0]
        for model in models:
            logL = float("-inf")
            try:
                model.fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
            except:
                logL = float("-inf")

            logD = 0
            M = 0
            for word in self.hwords:
                X_d, l_d = self.hwords[word]
                try:
                    logD += model.score(X_d, l_d)
                    M += 1
                except:
                    logD += 0
            if M>0:
                score = logL - (1 / M * logD)
            else:
                score = logL

            if score > best_score:
                best_score = score
                best_model = model
        return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        models = []
        for n in range(self.min_n_components, self.max_n_components+1):
            models.append(self.base_model(n))  # get the models

        if len(self.sequences) < 2:
            return None

        kf = KFold(n_splits=2) # define the Kfold
        indexes = kf.split(self.sequences)  # store the splitted indexes in a var         
        
        best_score = float("-inf")
        best_model = models[0]

        for model in models:
            score = 0
            count = 0
            for train_idx, test_idx in indexes:
                X_train, l_train = combine_sequences(train_idx, self.sequences)
                X_test, l_test = combine_sequences(test_idx, self.sequences) # get the data
                try:
                    model.fit(X_train, l_train)
                    score += model.score(X_test, l_test) # add the test result to score
                    count += 1 # add 1 to count in order to calc the avg
                except:
                    score += 0 # add nothing to score if model failed to fit the data
            if count>0:
                score /= count
            else:
                score = float("-inf")

            if score > best_score:
                best_score = score
                best_model = model
        return best_model

        
