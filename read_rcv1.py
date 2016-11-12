import os
import nltk
import xml.etree.ElementTree as ET

from collections import defaultdict, Counter
from dateutil import parser as dateparser

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

    def __init__(self, path):

        self.path = path
        doc_root = ET.parse(path).getroot()
        self.doc_id = doc_root.attrib['itemid']
        self.date = dateparser.parse(doc_root.attrib['date']).date()
        self.headline = doc_root.find('headline').text

        self.text = [sentence.text for sentence in doc_root.find('text')]


    def __str__(self):
        doc_id_and_date = "Document {doc_id} was published on {date}".format(doc_id=self.doc_id, date=self.date)
        headline = "Headline: {headline}".format(headline=self.headline)

        sentences = "Article Text: \n{text}".format(text="\n".join(self.text))

        return "\n".join([doc_id_and_date, headline, sentences])

    def headline_vocab(self):

        if self._headline_vocab:
            return self._headline_vocab

        tokened_headline = nltk.word_tokenize(self.headline)
        self._headline_vocab = Counter(tokened_headline)
        return self._headline_vocab

    def text_vocab(self):
        if self._text_vocab:
            return self._text_vocab

        self._text_vocab = nltk.FreqDist()
        for sentence in self.text:
           tokened_sentence = nltk.word_tokenize(sentence)
           self._text_vocab += Counter(tokened_sentence)

        return self._text_vocab



def read_rcv1_docs():

    date_dirs = [date_dir for date_dir in os.listdir(RCV1_DIR) if date_dir.startswith('199')]

    article_paths = {}
    for date in date_dirs[:3]:
        date_path = os.path.join(rcv1_dir, date)
        article_paths[date] = [os.path.join(date_path, article) for article in os.listdir(date_path) if article.endswith('.xml')]


    articles_by_date = defaultdict(list)
    for date, paths in article_paths.iteritems():
        for path in paths:
            articles_by_date[date].append(RCV1_doc(path))
        print date


    return articles_by_date


def create_training_splits():

    date_dirs = [date_dir for date_dir in os.listdir(RCV1_DIR) if date_dir.startswith('199')]

    article_paths = {'train': list(), 'val': list(), 'test': list()}

    for date in date_dirs:
        date_path = os.path.join(RCV1_DIR, date)
        count = 0
        for article_file_name in os.listdir(date_path):
            if not article_file_name.endswith('.xml'):
                continue

            article_path = os.path.join(date_path, article_file_name)
            if count < NUM_TRAIN_DAILY:
                article_paths['train'].append(article_path)
            elif count < NUM_TRAIN_DAILY + NUM_VAL_DAILY:
                article_paths['val'].append(article_path)
            elif count < NUM_TRAIN_DAILY + NUM_VAL_DAILY + NUM_TRAIN_DAILY:
                article_paths['test'].append(article_path)

            count += 1

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




if __name__ == "__main__":




    create_training_splits()
    # articles_by_date = read_rcv1_docs()

    # text_vocab = Counter()
    # headline_vocab = Counter()
    # total_vocab = Counter()


    # for date, articles in articles_by_date.iteritems():
    #     tots = len(articles)
    #     count = 0
    #     for article in articles:
    #         text_vocab.update(article.text_vocab())
    #         headline_vocab.update(article.headline_vocab())
    #         total_vocab.update(article.text_vocab())
    #         total_vocab.update(article.headline_vocab())



    # print "headline most common\n", headline_vocab.most_common(10)
    # print "\ntext most common\n", text_vocab.most_common(10)
    # print "\ntotal_most_common\n", total_vocab.most_common(10)




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
