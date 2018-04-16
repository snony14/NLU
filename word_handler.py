from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

fo = open("sent.txt", 'r')
sentences = [word_tokenize(line) for line in fo]

word_dict = dict()
for sent in sentences:
    for word in sent:
        if not word in word_dict:
            word_dict[word] = True

#creating a one-hot encoding of each observed words
ttl_words = len(word_dict)
print(ttl_words)
#one_hot_mat = np.zeros((ttl_words, ttl_words))
#np.fill_diagonal(one_hot_mat, 1)
np.save('one_hot_mat.npy', word_dict)
fo.close()
