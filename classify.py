#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter

import sys
import nltk
import codecs

from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
#from nltk.corpus import stopwords
from nltk.data import load

kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """
        #self.stop = set(stopwords.words("english"))
        # tagdict = load('help/tagsets/upenn_tagset.pickle')
        # print(tagdict.keys())

        None

    def features(self, text):
        d = defaultdict(int)
        wordcount=0
        totletters =0


        for ii in kTOKENIZER.tokenize(text):
           #if (not (ii in self.stop)):
            d[morphy_stem(ii)] += 1
            wordcount +=1
            totletters+=len(text)


        # for i in range(len(kTOKENIZER.tokenize(text))-1):
        #     d[((text[i]),(text[i+1]))] += 1


        for letter in 'aeiouv':
            d["count({})".format(letter)] = text.lower().count(letter)
            d["has({})".format(letter)] = (letter in text.lower())

        d["wordcount"]=wordcount
        d["avg_chars"] = totletters/wordcount

        # if(text.endswith('?')):
        #     d["quesn"]+=1


        return d

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None, help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()
    
    # Read in training data
    train = DictReader(trainfile, delimiter='\t')
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
         else:
            dev_train.append((feat, ii['cat'])) #all other lines are in training data -> remaining 80%
        full_train.append((feat, ii['cat']))    #entire 100%

    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)   #train on the dev_train to learn the features and their weights

    right = 0
    total = len(dev_test) #number of lines being tested
    for ii in dev_test:
        prediction = classifier.classify(ii[0])     #classify the text field in the dev_test file
        if prediction == ii[1]:
            right += 1              #no of correct classifications
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))
    #classifier.show_most_informative_features()

    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test) #train on the entire test data, TA will pass separate test data

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})
