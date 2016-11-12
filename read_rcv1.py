import os
import nltk
import xml.etree.ElementTree as ET

from collections import defaultdict, Counter
from dateutil import parser as dateparser




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


        self.headline_vocab()
        self.text_vocab()

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

    rcv1_dir = "rcv1"
    date_dirs = [date_dir for date_dir in os.listdir(rcv1_dir) if date_dir.startswith('199')]

    article_paths = {}
    for date in date_dirs[:1]:
        date_path = os.path.join(rcv1_dir, date)
        article_paths[date] = [os.path.join(date_path, article) for article in os.listdir(date_path) if article.endswith('.xml')]


    articles_by_date = defaultdict(list)
    for date, paths in article_paths.iteritems():
        for path in paths:
            articles_by_date[date].append(RCV1_doc(path))


    return articles_by_date





if __name__ == "__main__":


    articles_by_date = read_rcv1_docs()

    text_vocab = Counter()
    headline_vocab = Counter()
    total_vocab = Counter()


    for date, articles in articles_by_date.iteritems():
        tots = len(articles)
        count = 0
        for article in articles:
            text_vocab.update(article.text_vocab())
            headline_vocab.update(article.headline_vocab())
            total_vocab.update(article.text_vocab())
            total_vocab.update(article.headline_vocab())



    print "headline most common\n", headline_vocab.most_common(10)
    print "\ntext most common\n", text_vocab.most_common(10)
    print "\ntotal_most_common\n", total_vocab.most_common(10)




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
