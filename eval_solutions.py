import nltk
from nltk.translate.bleu_score import modified_precision
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


import read_rcv1 as rr




def eval_first_sent(data=None):

    if data == None:
        data = rr.get_split_data()


    stemmer = PorterStemmer()

    bleu_1_scores = list()
    bleu_2_scores = list()
    bleu_stem_scores = list()
    bleu_stem2_scores = list()
    count = 0
    for article in data:
        ref = article.headline
        hypothesis = article.text[0][:10]
        hyp2 = article.text[0]
        ref_stem = [stemmer.stem(word) for word in article.headline]
        hyp_stem = [stemmer.stem(word) for word in article.text[0]]

        bleu1 = float(modified_precision([ref], hypothesis, n=1))
        bleu2 = float(modified_precision([ref], hyp2, n=1))
        bleu_stem = float(modified_precision([ref_stem], hyp_stem[:10], n=1))
        bleu2_stem = float(modified_precision([ref_stem], hyp_stem, n=1))

        bleu_1_scores.append(bleu1)
        bleu_2_scores.append(bleu2)
        bleu_stem_scores.append(bleu_stem)
        bleu_stem2_scores.append(bleu2_stem)

        count += 1.
        rr.update_progress(count / len(data))

    bleu1_avg = sum(bleu_1_scores) / len(bleu_1_scores)
    print bleu1_avg
    bleu2_avg = sum(bleu_2_scores) / len(bleu_2_scores)
    print bleu2_avg
    bleu1_stem_avg = sum(bleu_stem_scores) / len(bleu_stem_scores)
    print bleu1_stem_avg
    bleu2_stem_avg = sum(bleu_stem2_scores) / len(bleu_stem2_scores)
    print bleu2_stem_avg

    return bleu1_avg, bleu2_avg, bleu1_stem_avg, bleu2_stem_avg


def get_tf_idf(data=None):

    if data == None:
        data = rr.get_split_data()


    corpus = list() # a list of documents as strings
    for article in data:
        corpus.append(" ".join(" ".join(sent) for sent in article.text))

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(corpus)

    feature_names = tfidf.get_feature_names()


    bleu1_scores = list()
    count = 0.
    for doc_row in range(len(corpus)):
        feature_indicies = tfidf_matrix[doc_row,:].nonzero()[1]

        word_scores = [(feature_names[x], tfidf_matrix[doc_row, x]) for x in feature_indicies]

        sorted_word_scores = sorted(word_scores, key=lambda t: -t[1])

        top_ten_words = [str(word[0]) for word in sorted_word_scores[:10]]

        headline_lowered = [word.lower() for word in data[doc_row].headline]

        bleu1 = float(modified_precision([headline_lowered], top_ten_words, n=1))

        bleu1_scores.append(bleu1)
        count += 1.
        rr.update_progress(count / len(corpus))

    bleu1_avg = sum(bleu1_scores) / len(bleu1_scores)
    print bleu1_avg
    return bleu1_avg



