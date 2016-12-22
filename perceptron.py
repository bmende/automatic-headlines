import read_rcv1 as read
import math
import operator
import random
import itertools
from collections import defaultdict

weights = defaultdict(float)
weight_feats = defaultdict(float)
normalizer = float(0)


def main(path,mode):

    if mode == 'train':
        if type(path) == str:
            docs = read.get_split_data(path)
        else:
            docs = path
        count = 0.
        for doc in docs[:150]:
            count += 1.			#temporary to keep a count of the documents processed
            for index in range(doc.article_length):
                feature_vec, outcome = doc.get_local_feature(index)
                train(feature_vec,outcome)
            read.update_progress(count / len(docs))		#temporary
        print "count is", count
        return
    elif mode == 'test':
        prob = {}
        if type(path) == str:
            doc = read.get_one_article(path)
        else:
            doc = path
        for index in range(doc.article_length):
            feature_vec, outcome = doc.get_local_feature(index)
            prob[doc.text_pos[index][0]] = test(feature_vec)
        topn = sorted(prob.items(), key=operator.itemgetter(0), reverse=True)[:10]
        return topn

def train(feature_vec,outcome,learning_rate=0.001):			#have ignored validation set for now and set learning_rate arbitrarily

    output = predict(feature_vec)
    error = outcome - output
    for k in feature_vec.keys():
        weights[k] += learning_rate*error

def predict(feature_vec):

    expo_sum = 0
    for key,value in feature_vec.iteritems():
        if weights[key] == 0.0:
            weights[key] = random.uniform(0, 1)
        weight_feats[key] = weights[key]*value
    normalize = normalizer
    normalize += max(weight_feats.values())
    #for keys in weight_feats.iterkeys():
    #    weight_feats[key] -= normalize
    expo_sum = sum(math.exp(weight_feats[key] - normalize) for key in feature_vec)
#    for key in feature_vec.keys():
#        expo_sum += math.exp(weight_feats[key] - normalize)
    output = math.log(expo_sum)/expo_sum
    return output

def test(feature_vec):
    output = predict(feature_vec)
    return output

if __name__ == "__main__":

    main('data/train_sample365.split','train')		#small files that I made created with 1 file a day instead of 100
    topn = main('data/rcv1/19961021/131576newsML.xml','test')
    print "topn",topn
