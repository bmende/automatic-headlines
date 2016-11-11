import os
import nltk

import xml.etree.ElementTree as ET

from dateutil import parser as dateparser




class RCV1_doc:

    def __init__(self, path):

        self.path = path
        doc_root = ET.parse(path).getroot()
        self.doc_id = doc_root.attrib['itemid']
        self.date = dateparser.parse(doc_root.attrib['date']).date()
        self.headline = doc_root.find('headline').text

        self.text = [sentence.text for sentence in doc_root.find('text')]

    def __str__(self):
        ret_str = "Document {doc_id} was published on {date} \n".format(doc_id=self.doc_id, date=self.date)
        ret_str += self.headline + "\n"

        sentences = "\n".join(self.text)

        ret_str += sentences

        return ret_str







if __name__ == "__main__":

    rcv1_dir = "rcv1"
    date_dirs = [date_dir for date_dir in os.listdir(rcv1_dir) if date_dir.startswith('199')]

    article_paths = {}
    for date in date_dirs:
        date_path = os.path.join(rcv1_dir, date)
        article_paths[date] = [os.path.join(date_path, article) for article in os.listdir(date_path) if article.endswith('.xml')]

    print date, article_paths[date][:1], "\n"
    first_doc_last_day = RCV1_doc(article_paths[date][0])
    print str(first_doc_last_day)

    print len(first_doc_last_day.text)
