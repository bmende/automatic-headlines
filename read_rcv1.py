import os, sys, pickle
import nltk
import xml.etree.ElementTree as ET

from collections import defaultdict, Counter
from dateutil import parser as dateparser
from itertools import chain

DATA_DIR = "data"
RCV1_DIR = os.path.join(DATA_DIR, "rcv1")


NUM_TRAIN_DAILY = 100
NUM_VAL_DAILY = 50
NUM_TEST_DAILY = 50
# 365 is number of article directories, one for each day
NUM_TRAIN = 365 * NUM_TRAIN_DAILY # 36,500
NUM_VAL = 365 * NUM_VAL_DAILY # 18,250
NUM_TEST = 365 * NUM_TEST_DAILY # 18,250



class RCV1_doc:

    _headline_vocab = None
    _text_vocab = None
    _first_sent_vocab = None

    def __init__(self, path):

        self.path = path
        doc_root = ET.parse(path).getroot()
        self.doc_id = doc_root.attrib['itemid']
        self.date = dateparser.parse(doc_root.attrib['date']).date()

        self.headline = nltk.word_tokenize(doc_root.find('headline').text or [""])
        self.headline_set = set(self.headline) # to find if words are in the headline, this is faster

        self.text = [nltk.word_tokenize(sentence.text) for sentence in doc_root.find('text')]

        self.headline_pos = self.get_headline_pos()
        self.text_pos = self.get_text_pos()


    @staticmethod
    def has_headline(path):
        doc_root = ET.parse(path).getroot()
        return doc_root.find('headline').text is not None

    def __str__(self):
        doc_id_and_date = "Document {doc_id} was published on {date}".format(doc_id=self.doc_id, date=self.date)
        headline = "Headline: {headline}".format(headline=self.headline)

        sentences = "Article Text: \n{text}".format(text="\n".join(self.text))

        return "\n".join([doc_id_and_date, headline, sentences])

    def headline_vocab(self):

        if self._headline_vocab:
            return self._headline_vocab
        self._headline_vocab = Counter(self.headline)
        return self._headline_vocab

    def first_sent_vocab(self):

        if self._first_sent_vocab:
            return self._first_sent_vocab
        self._first_sent_vocab = Counter(self.text[0])
        return self._first_sent_vocab

    def text_vocab(self):
        if self._text_vocab:
            return self._text_vocab

        self._text_vocab = Counter()
        for sentence in self.text:
           self._text_vocab.update(Counter(sentence))

        return self._text_vocab

    def get_text_pos(self):
        text_pos = list()
        for sent in self.text:
            text_pos.extend(nltk.pos_tag(sent))

        return text_pos

    def get_headline_pos(self):
        return nltk.pos_tag(self.headline)


    def get_local_feature(self, word_index):
        max_index = len(self.text_pos)
        assert word_index <= max_index

        word, word_pos = self.text_pos[word_index]
        word_sentence = 0
        word_i = word_index - len(self.text[0])
        while word_i >= 0:
            word_sentence += 1
            word_i -= len(self.text[word_sentence])

        in_headline = word in self.headline_set
        local_feature = dict()
        local_feature[(in_headline, 'currword', word)] = 1
        local_feature[(in_headline, 'currword_pos', word_pos)] = 1
        local_feature[(in_headline, 'currword_sentence', word_sentence)] = 1

        # word context is 2 before, and 2 after
        prev_word, prev_word_pos = self.text_pos[word_index-1] if word_index > 0 else (None, None)
        post_word, post_word_pos = self.text_pos[word_index+1] if word_index < max_index else (None, None)

        local_feature[(in_headline, 'pre_bigram', word, prev_word)] = 1
        local_feature[(in_headline, 'post_bigram', word, post_word)] = 1
        local_feature[(in_headline, 'pre_pos', prev_word_pos)] = 1
        local_feature[(in_headline, 'post_pos', post_word_pos)] = 1

        return local_feature





def get_split_data(split_path_file='data/train36500.split'):


    with open(split_path_file, 'r') as splits:
        split_paths = [path.strip() for path in splits]


    rcv1_articles = list()
    tots = 10#len(split_paths)
    count = 0.
    for path in split_paths[:tots]:
            rcv1_articles.append(RCV1_doc(path))
            count += 1
            update_progress(count / tots)


    return rcv1_articles

def get_one_article(article_path='data/rcv1/19961021/131576newsML.xml'):

    return RCV1_doc(article_path)


def create_training_splits():

    date_dirs = [date_dir for date_dir in os.listdir(RCV1_DIR) if date_dir.startswith('199')]

    article_paths = {'train': list(), 'val': list(), 'test': list()}

    total_count = 0
    for date in date_dirs:
        date_path = os.path.join(RCV1_DIR, date)
        count = 0
        for article_file_name in os.listdir(date_path):
            if not article_file_name.endswith('.xml'):
                continue

            article_path = os.path.join(date_path, article_file_name)
            if not RCV1_doc.has_headline(article_path):
                print article_path
                continue # cant have articles with no headline!
            if count < NUM_TRAIN_DAILY:
                article_paths['train'].append(article_path)
            elif count < NUM_TRAIN_DAILY + NUM_VAL_DAILY:
                article_paths['val'].append(article_path)
            elif count < NUM_TRAIN_DAILY + NUM_VAL_DAILY + NUM_TEST_DAILY:
                article_paths['test'].append(article_path)
            else:
                break

            count += 1
            total_count += 1
            update_progress(float(total_count) / float(NUM_TRAIN + NUM_VAL + NUM_TEST))

    train_splits = open(os.path.join(DATA_DIR, 'train{size}.split'.format(size=NUM_TRAIN)), 'w')
    val_splits = open(os.path.join(DATA_DIR, 'val{size}.split'.format(size=NUM_VAL)), 'w')
    test_splits = open(os.path.join(DATA_DIR, 'test{size}.split'.format(size=NUM_TEST)), 'w')


    for article_path in article_paths['train']:
        train_splits.write(article_path + '\n')

    for article_path in article_paths['val']:
        val_splits.write(article_path + '\n')

    for article_path in article_paths['test']:
        test_splits.write(article_path + '\n')

    train_splits.close()
    val_splits.close()
    test_splits.close()




# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}%    {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__ == "__main__":




    #create_training_splits()
    print "getting data"
    train_articles = get_split_data('data/val{size}.split'.format(size=NUM_VAL))
    print "data got"

    # text_vocab = Counter()
    # headline_vocab = Counter()
    # first_sent_vocab = Counter()
    # total_vocab = Counter()


    # count = 0.
    # total = len(train_articles)
    # print "getting stats"
    # for article in train_articles:
    #     text_vocab.update(article.text_vocab())
    #     headline_vocab.update(article.headline_vocab())
    #     total_vocab.update(article.text_vocab())
    #     total_vocab.update(article.headline_vocab())
    #     first_sent_vocab.update(article.first_sent_vocab())
    #     count += 1
    #     update_progress(count / total)



    # with open('hv', 'r') as hv:
    #     headline_vocab = pickle.load(hv)
    # with open('tev', 'r') as tev:
    #    text_vocab = pickle.load(tev)
    # with open('totv', 'r') as totv:
    #    total_vocab = pickle.load(totv)
    # with open('firstv', 'r') as firstv:
    #     first_sent_vocab = pickle.load(firstv)
    # print "headline most common\n", headline_vocab.most_common(10)
    # print "\ntext most common\n", text_vocab.most_common(10)
    # print "\ntotal_most_common\n", total_vocab.most_common(10)
    # print "\nfirst_sent_most_common\n", first_sent_vocab.most_common(10)

    # print "\ntotal number of words in articles, headlines, and combined"
    # print sum(text_vocab.values()), sum(headline_vocab.values()), sum(total_vocab.values())
    # print "size of vocabulary in articles, headlines, and combined"
    # print len(text_vocab.values()), len(headline_vocab.values()), len(total_vocab.values())

    # first_fd = nltk.FreqDist(first_sent_vocab)
    #first_fd.plot(25, title="25 Most common words in the first sentence")




    # rcv1_dir = "rcv1"
    # date_dirs = [date_dir for date_dir in os.listdir(rcv1_dir) if date_dir.startswith('199')]

    # article_paths = {}
    # for date in date_dirs:
    #     date_path = os.path.join(rcv1_dir, date)
    #     article_paths[date] = [os.path.join(date_path, article) for article in os.listdir(date_path) if article.endswith('.xml')]

    # print date, article_paths[date][:1], "\n"
    # first_doc_last_day = RCV1_doc(article_paths[date][0])
    # print str(first_doc_last_day)

    # print len(first_doc_last_day.text)
