"""\
------------------------------------------------------------
USE: python <PROGNAME> (options)
OPTIONS:
    -h : print this help message
    -s SCALE : use SCALE-value sentiment scale (SCALE in {3, 5}, default: 3)
    -m MODE : running mode MODE (MODE in {train, predict, eval}, default: predict)
    -i FILE : input phrase from file FILE
    -n FILE : Model file FILE
    -r FILE : Prediction result file FILE
    -f FLOAT : Use feature filter. FLOAT represent the number of feature tokens as a ratio relative to all the distinct tokens in the training set. (0.0-1.0) (training option)
    -p : Plot confusion matrix (eval option)
    -d : debug mode (enable feature rank dump file)

EXAMPLES:
    python <PROGNAME> -s 3 -m train -i train.tsv -n model_3classes.model -f 0.8
    python <PROGNAME> -s 3 -m predict -i dev.tsv -n model_3classes.model -r dev_predictions_3classes.tsv
    python <PROGNAME> -s 3 -m eval -i dev.tsv -r dev_predictions_3classes.tsv -p
------------------------------------------------------------\
"""

import getopt
import pickle
import string
import sys
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer

#==============================================================================
# Command line processing

class CommandLine:
    ''' CommandLine util.
        Parse command line params.
    '''

    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'hs:m:i:n:r:f:pd')
        opts = dict(opts)
        self.exit = True

        if '-h' in opts:
            self.printHelp()
            return

        if len(args) > 0:
            print("*** ERROR: no arg files - only options! ***", file=sys.stderr)
            self.printHelp()
            return

        if '-s' in opts:
            if opts['-s'] in ('3', '5'):
                self.scale = int(opts['-s'])
            else:
                warning = (
                    "*** ERROR: sentiment scale (opt: -s SCALE)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: 3 / 5"
                    )  % (opts['-s'])
                print(warning, file=sys.stderr)
                self.printHelp()
                return
        else:
            self.scale = 3

        if '-m' in opts:
            if opts['-m'] in ('train', 'predict', 'eval'):
                self.mode = opts['-m']
            else:
                warning = (
                    "*** ERROR: running mode (opt: -m MODE)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: train / predict / eval"
                    )  % (opts['-m'])
                print(warning, file=sys.stderr)
                self.printHelp()
                return
        else:
            self.mode = 'predict'

        if '-i' in opts:
            self.infile = opts['-i']
        else:
            print("*** ERROR: must specify phrase file (opt: -i FILE) ***",
                  file=sys.stderr)
            self.printHelp()
            return

        if '-n' in opts:
            self.model = opts['-n']
        else:
            if self.mode == 'train' or self.mode == 'predict':
                print("*** ERROR: must specify a model file (opt: -n FILE) ***",
                    file=sys.stderr)
                self.printHelp()
                return

        if '-r' in opts:
            self.predictfile = opts['-r']
        else:
            if self.mode == 'predict' or self.mode == 'eval':
                print("*** ERROR: must specify a predict output file (opt: -n FILE) ***",
                    file=sys.stderr)
                self.printHelp()
                return

        self.filter_features = '-f' in opts
        if '-f' in opts:
            self.trimming_ratio = float(opts['-f'])

        self.plot_confusion_matrix = '-p' in opts

        self.debug = '-d' in opts
            
        self.exit = False

    def printHelp(self):
        progname = sys.argv[0]
        progname = progname.split('/')[-1] # strip off extended path
        help = __doc__.replace('<PROGNAME>', progname, 1)
        print(help, file=sys.stderr)

#==============================================================================
# Common

class SentimentScales:
    ''' Model of different sentiment scales.
    '''

    scale_3 = [ 'negative', 'neutral' , 'positive' ]

    scale_5 = [ 'negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive' ]

class Phrase:
    ''' Model a Phrase.
    '''

    def __init__(self, id, tokens, sentiment):
        ''' Create a Phrase object with id, tokens, sentiment.

            Args:
                id (int): ID of the Phrase.
                tokens ([str]): A list of token strs.
                sentiment (int): Sentiment tag number (start from 0)
        '''
        self.id = id
        self.tokens = tokens
        self.sentiment = sentiment

    @classmethod
    def load_phrases(cls, filename):
        ''' Load phrases from a tsv file.

            Args:
                filename (str): Filename of the input tsv.

            Returns:
                A list of Phrase objects.
        '''
        phrases = []
        df = pd.read_csv(filename, index_col=0, delimiter='\t')

        parse_sentiment = 'Sentiment' in df.columns

        for index, row in df.iterrows():
            if parse_sentiment:
                phrase = Phrase(index, row['Phrase'].split(), row['Sentiment'])
            else:
                phrase = Phrase(index, row['Phrase'].split(), -1)
            phrases.append(phrase)
        return phrases

class Preprocessor:
    ''' Model a token pre-processor.
    '''

    def __init__(self, phrases):
        ''' Create a phrase pre-processor.

            Args:
                phrases ([Parse]): List of Phrases to be processed.
        '''
        self.phrases = phrases
        
        # Init stopwords
        self.stoplist = stopwords.words('english')
        self.stoplist.extend(string.punctuation.replace('!', ''))
        self.stoplist.extend([ '\'s', '``', '\'\'', '...', '--', 'n\'t', '\'d' ])
        self.stoplist = set(self.stoplist)

        # Init Stemmer (removed)
        # self.stemmer = SnowballStemmer('english')

        # Init sentiment scale mapping
        self.scale_5_3 = {
            0:0,
            1:0,
            2:1,
            3:2,
            4:2
        }

    def preprocessing(self):
        ''' Apply pre-processing to the Phrases
        '''
        for phrase in self.phrases:
            # Capitalisation
            phrase.tokens = [i.lower() for i in phrase.tokens ]

            # Stemming (Removed)
            # phrase.tokens = [self.stemmer.stem(i) for i in phrase.tokens ]

            # Stopword removal
            phrase.tokens = [i for i in phrase.tokens if i not in self.stoplist ]

        return self.phrases

    def to_scale_3(self):
        ''' Convert sentiment scale from 5 to 3
        '''
        for phrase in self.phrases:
            if phrase.sentiment < 0:
                # Not available
                continue
            phrase.sentiment = self.scale_5_3[phrase.sentiment]
        return self.phrases

class FeatureProcessor:
    ''' Model a Feature Processor.
        Used to analysis and extract 'feature' tokens from the training set, and filter the tokens in phrase based on feature tokens.
    '''

    def __init__(self, features=[], trimming_ratio=0.2, dump_sorted_terms_biases=False):
        ''' Create a Feature Processor.

        Args:
            features ([str]): Init FeatureProcessor with a list of feature tokens.
            trimming_ratio (float [0.0, 1.0]): The ratio of the trimmed 'feature' token list to all tokens in the training set when extracting the "feature" token.
                                               0.0 implies that the feature list will be an empty list, 1.0 means that the feature list will be all the tokens in training sets.
        '''
        self.features = features
        self.trimming_ratio = trimming_ratio

        # Dump sorted_terms_biases (Debug)
        self.dump_sorted_terms_biases = dump_sorted_terms_biases

    def extract_features(self, phrases, scale):
        ''' Analysis and Extract feature tokens from a list of phrases.

        Args:
            phrases ([Phrases]): List of phrases for analysis and extraction.
            scale (int): The sentiment scale used to analysis.
        '''
        # Count the number of times each token appears in different sentiment
        token_sentiment_counts_dict = dict()
        for phrase in phrases:
            sentiment = phrase.sentiment
            for token in phrase.tokens:
                if token not in token_sentiment_counts_dict:
                    token_sentiment_counts_dict[token] = [0] * scale
                token_sentiment_counts_dict[token][sentiment] += 1

        # Filter out tokens with too few occurrences
        # token_sentiment_counts = [(term, sentiment_counts) for term, sentiment_counts in token_sentiment_counts_dict.items() if sum(sentiment_counts) > scale]

        # Calculate the sentiment bias of each token
        sentiment_biases = []
        for token, sentiment_counts in token_sentiment_counts_dict.items():
            sentiment_bias = 0

            # Calulate sentiment_bias 
            # Neutral sentiment weights 0, Positive sentiment weights positive value, Negative sentiment weights negative value
            for i in range(scale):
                factor = i - (scale - 1) / 2
                sentiment_bias += sentiment_counts[i] * factor

            # Normalization
            # sentiment_bias = sentiment_bias / sum(sentiment_counts)
            # sentiment_bias = sentiment_bias * sentiment_bias

            sentiment_biases.append((token, sentiment_bias))
        
        # Sort list based on sentiment bias
        sorted_terms_biases = sorted(sentiment_biases, key=lambda item: item[1] * item[1], reverse=True)

        # Dump sorted_terms_biases (Debug)
        if self.dump_sorted_terms_biases:
            f = open('sorted_terms_biases.txt', 'w')
            for b in sorted_terms_biases:
                f.write(str(b[1]) + '\t' + b[0] + '\n')
            f.close()
        
        # Trim features list
        trim_length = int(len(sentiment_biases) * self.trimming_ratio)
        trimed_sorted_terms_biases = sorted_terms_biases[:trim_length]
        self.features = [i[0] for i in trimed_sorted_terms_biases]
        return self.features

    def filter_features(self, phrases):
        ''' Apply feature filter to a list of phrases.

        Args:
            phrases ([Phrases]): List of phrases that need to be filtered.
        '''
        for phrase in phrases:
            phrase.tokens = [i for i in phrase.tokens if i in self.features ]
        return phrases

#==============================================================================
# Training

class CorpusMeta:
    ''' Model the metadata for a list of phrases. Used to produce trained models.
        Fields:
            self.sentiment_scale ([str]): A list of scale names in the sentiment scale.
            self.phrase_count (int): Number of phrases in the phrase set.
            self.sentiment_phrase_counts ([int]): The number of phrases in each sentiment. The length of the list is the same as the length of sentiment_scale.
            self.sentiment_token_counts ([int]): The total number of tokens in each sentiment. The length of the list is the same as the length of sentiment_scale.
            self.relative_sentiment_token_counts ([{str: int}]): The number of occurrences of each token in each sentiment.
            self.vocabulary_count (int): The number of distinct tokens.
    '''

    def __init__(self, phrases, sentiment_scale):
        ''' Create a CorpusMeta object.

        Args:
            phrases ([Phrase]): List of phrases.
            sentiment_scale ([str]): A list of scale names in the sentiment scale.
            feature [str]: List of feature tokens used in this phrase set.
        '''
        self.sentiment_scale = sentiment_scale
        self.phrase_count = len(phrases)
        self.sentiment_phrase_counts = self.__compute_sentiment_phrase_counts(phrases, sentiment_scale)
        self.sentiment_token_counts = self.__compute_sentiment_token_counts(phrases, sentiment_scale)
        self.relative_sentiment_token_counts = self.__compute_relative_sentiment_token_counts(phrases, sentiment_scale)
        self.vocabulary_count = self.__compute_vocabulary_count(phrases)

    def __compute_sentiment_phrase_counts(self, phrases, scale):
        phrase_counter = Counter()

        for phrase in phrases:
            sentiment = phrase.sentiment
            phrase_counter[sentiment] += 1

        phrase_counts = []
        for i in range(len(scale)):
            phrase_counts.append(phrase_counter[i])

        return phrase_counts

    def __compute_sentiment_token_counts(self, phrases, scale):
        token_counter = Counter()

        for phrase in phrases:
            sentiment = phrase.sentiment
            token_counter[sentiment] += len(phrase.tokens)

        phrase_counts = []
        for i in range(len(scale)):
            phrase_counts.append(token_counter[i])

        return phrase_counts

    def __compute_relative_sentiment_token_counts(self, phrases, scale):
        relative_counters = []
        for i in range(len(scale)):
            relative_counters.append(Counter())

        for phrase in phrases:
            sentiment = phrase.sentiment
            for token in phrase.tokens:
                relative_counters[sentiment][token] += 1
        
        relative_counts = []
        for i in range(len(scale)):
            count_dict = dict(relative_counters[i])
            relative_counts.append(count_dict)
        
        return relative_counts
    
    def __compute_vocabulary_count(self, phrases):
        vocabulary = set()
        for phrase in phrases:
            vocabulary.update(phrase.tokens)
        return len(vocabulary)
            
#==============================================================================
# Pridicting

class Predictor:
    ''' Model a Predictor. Predict sentiment based on a CorpusMeta.
    '''

    def __init__(self, corpus_meta):
        ''' Create a Predictor object based on a CorpusMeta.

            Args:
                corpus_meta (CorpusMeta): CorpusMeta used for prediction.
        '''
        self.corpus_meta = corpus_meta

    def predict_likelihoods(self, phrases):
        ''' Predict the likelihood to each sentiment for phrases.

            Args:
                phrases ([Phrase]): Phrases that need to be predicted sentiment.

            Returns:
                ([[float]]) A list of prediction results, each prediction is a list of likelihoods corresponding to each sentiment.
        '''
        corpus_meta = self.corpus_meta
        results = []
        for phrase in phrases:
            likelihoods = []
            for i in range(len(corpus_meta.sentiment_scale)):
                sentiment_phrase_count = corpus_meta.sentiment_phrase_counts[i]
                if corpus_meta.sentiment_phrase_counts[i] > 0:
                    # prior probability
                    prob = sentiment_phrase_count / corpus_meta.phrase_count

                    # relative likelihoods
                    relative_token_counts = corpus_meta.relative_sentiment_token_counts[i]
                    sentiment_token_count = corpus_meta.sentiment_token_counts[i]
                    smoothed_sentiment_token_count = sentiment_token_count + corpus_meta.vocabulary_count
                    for token in phrase.tokens:
                        if token in relative_token_counts:
                            relative_token_count = relative_token_counts[token]
                        else:
                            relative_token_count = 0
                        smoothed_relative_token_count = relative_token_count + 1
                        token_prob = smoothed_relative_token_count / smoothed_sentiment_token_count
                        prob *= token_prob
                    likelihoods.append(prob)
                else:
                    likelihoods.append(0)
            results.append(likelihoods)
        return results

    def predict(self, phrases):
        ''' Predict the sentiment for phrases.

            Args:
                phrases ([Phrase]): Phrases that need to be predicted sentiment.

            Returns:
                ([int]) A list of prediction results, each result is the index of the sentiment.
        '''
        results = []
        likelihoods_list = self.predict_likelihoods(phrases)
        for likelihoods in likelihoods_list:
            argmax = max(range(len(likelihoods)), key=lambda x: likelihoods[x])
            results.append(argmax)
        return results

#==============================================================================
# Evaluation

class Evaluator:
    ''' Model a Evaluator. Evaluate the predicted results.
    '''

    def __init__(self, phrases, predict_result_dict, scale):
        ''' Create a Evaluator object.

            Args:
                phrases (Phrases): List of phrases containing true results.
                predict_result_dict ({int, int}): A ID-sentiment dictionary of predicted results.
        '''
        self.accuracy = None
        self.phrases = phrases
        self.predict_result_dict = predict_result_dict
        self.scale = scale

        cf_matrix = [[0] * scale for i in range(scale)]
        for phrase in phrases:
            id = phrase.id
            if id in predict_result_dict:
                cf_matrix[phrase.sentiment][predict_result_dict[id]] += 1
            else:
                print('missing prediction', id)
        
        self.cf_matrix = cf_matrix

    def compute_accuracy(self):
        ''' Compute the accuracy of prediction.
            
            Return:
                Accuracy of prediction.
        '''
        if self.accuracy != None:
            return self.accuracy
        accurate_count = sum([self.cf_matrix[i][i] for i in range(self.scale)])
        total_count = sum([sum(self.cf_matrix[i]) for i in range(self.scale)])
        accuracy = accurate_count / total_count
        return accuracy

    def plot_confusion_matrix(self, target_names=None, title='Confusion matrix', cmap=None):
        ''' Show confusion matrix.

            Args:
                target_names ([str]): Target names used for plotting.
                title (str): The title of the figure.
                cmap (str or matplotlib Colormap): Color map of the heatmap.
        '''
        cf_matrix = self.cf_matrix
        scale = self.scale
        
        # accuracy = self.compute_accuracy()
        # misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        # plt.figure(figsize=(8, 6))
        # plt.figure(figsize=(4, 3.5))
        plt.figure(figsize=(3, 2.5))
        plt.imshow(cf_matrix, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = range(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
        else:
            tick_marks = range(scale)
            plt.xticks(tick_marks, tick_marks)
            plt.yticks(tick_marks, tick_marks)

        textcolor_thresh = max([max(i) for i in cf_matrix]) / 2
        for i in range(scale):
            for j in range(scale):
                plt.text(j, i, "{:,}".format(cf_matrix[i][j]),
                        horizontalalignment="center",
                        color="white" if cf_matrix[i][j] > textcolor_thresh else "black")

        plt.ylabel('True label')
        # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

#==============================================================================
# Main

def main(config):
    # load phrases
    phrases = Phrase.load_phrases(config.infile)

    # pre-processing phrases
    preprocessor = Preprocessor(phrases)
    if config.mode == 'train' or config.mode == 'predict':
        preprocessor.preprocessing()

    # rescale sentiment
    sentiment_scale = SentimentScales.scale_5
    if config.scale == 3:
        sentiment_scale = SentimentScales.scale_3
        preprocessor.to_scale_3()

    # run mode
    if config.mode == 'train':
        # extract features and apply feature filter
        if config.filter_features:
            featureProcessor = FeatureProcessor(trimming_ratio=config.trimming_ratio, dump_sorted_terms_biases=config.debug)
            featureProcessor.extract_features(phrases, config.scale)
            featureProcessor.filter_features(phrases)

        # train
        corpus_meta = CorpusMeta(phrases, sentiment_scale)

        # save model
        with open(config.model, 'wb') as f:
            pickle.dump(corpus_meta, f)
    elif config.mode == 'predict':
        # load model
        with open(config.model, 'rb') as f:
            corpus_meta = pickle.load(f)

        # verify
        if len(corpus_meta.sentiment_scale) != config.scale:
            raise Exception('Error : sentiment scale not match')

        # predict result
        predictor = Predictor(corpus_meta)
        predicted_results = predictor.predict(phrases)

        # save result
        with open(config.predictfile, 'w') as f:
            f.write('SentenceId\tSentiment\n')
            for i in range(len(predicted_results)):  
                f.write(str(phrases[i].id) + '\t' + str(predicted_results[i]) + '\n')
    elif config.mode == 'eval':
        # load predictions
        df = pd.read_csv(config.predictfile, index_col=0, delimiter='\t')
        predict_dict = dict()
        for index, row in df.iterrows():
            predict_dict[index] = row['Sentiment']

        # evaluation
        evaluationer = Evaluator(phrases, predict_dict, config.scale)
        accuracy = evaluationer.compute_accuracy()
        print('accuracy:', accuracy)
        if config.plot_confusion_matrix:
            evaluationer.plot_confusion_matrix()

if __name__ == "__main__":
    config = CommandLine()
    if config.exit:
        sys.exit(0)
    main(config)
