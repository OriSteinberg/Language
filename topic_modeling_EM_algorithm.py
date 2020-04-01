import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import itertools


def main(dev_set_file):
    num_of_topics = 9
    threshold = 0.01
    model = EM(dev_set_file, num_of_topics)
    print(len(model.corpus_freq))
    model.run(threshold)

    with open('out.txt', 'w') as f:
        for i, article in enumerate(model.w_t_i):
            label = np.argmax(article)
            f.write('{}\t{}\n'.format(i, label))
    confusion_matrix(model, 'file')
    accu(model)
    print('finish')


def accu(model):
    T2I, I2T = read_topics()
    with open('algo_class.txt', 'r') as f:
        counter = 0.
        all = 0.
        my_class = [3, 3, 0, 2, 1, 5, 4, 1, 4]
        for line, labels in zip(f, model.labels):
            all += 1
            i, topic = line.split()
            topic = I2T[my_class[int(topic)]]
            if topic in labels:
                counter += 1
    print('{}  {}  {}'.format(counter, all, counter/all))


def confusion_matrix(model, file_or_model):
    topics, I2T = read_topics()
    cm = np.zeros([model.topic_num, model.topic_num], dtype=np.int32)
    if file_or_model == 'model':
        for article, labels in zip(model.w_t_i, model.labels):
            line = np.argmax(article)
            for label in labels[1:]:
                cm[line, topics[label]] += 1
    elif file_or_model == 'file':
        with open('algo_class.txt', 'r') as f:
            for line, labels in zip(f, model.labels):
                _, topic = line.split()
                for label in labels[1:]:
                    cm[int(topic), topics[label]] += 1

    print(cm)
    names = list(topics.keys())
    classes = list(topics.values())
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, classes, names)


def plot_confusion_matrix(cm, pred_classes, real_topics, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(pred_classes))
    plt.xticks(tick_marks, real_topics, rotation=45)
    plt.yticks(tick_marks, pred_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted topics')
    plt.xlabel('True topics')


def read_topics():
    with open('topics.txt', 'r') as f:
        topics = []
        for line in f:
            line = line.split()
            if len(line) > 0:
                topics.append(line[0])
    topic_dict = {topic: i for i, topic in enumerate(topics)}
    I2T = {i: topic for i, topic in enumerate(topics)}
    return topic_dict, I2T


class EM():
    def __init__(self, dev_file, topics_len):
        # Paramrters
        self.rare_w_th = 3 
        self.k = 10
        self.epsilon = 0.00001
        self.lmbda = 0.1
        self.topic_num = topics_len

        # set Corpus matrixes
        self.read_article(dev_file)  # articles_without_rare, corpus_freq, labels,
        self.n_t = [len(article) for article in self.articles_without_rare]
        self.n_t_k = [Counter(article) for article in self.articles_without_rare]
        self.N = len(self.articles_without_rare)
        self.a_i = np.random.uniform(0, 1, [topics_len, 1])
        self.p_i_k = []
        for i in range(topics_len):
            self.p_i_k.append({word: np.random.random() for word in self.corpus_freq.keys()})  # row - i, colu - k
        self.init()
        self.total_num_of_word = sum(n for n in self.n_t)

    def E_step(self):
        # for every article we have a list of word_freq in each topic -> # row - t, colu - i
        z_t_i = [[np.log(self.a_i[ix]) + sum(self.n_t_k[t][word] * np.log(freq) for word, freq in self.p_i_k[ix].items())
                 for ix in range(self.topic_num)] for t in range(self.N)]
        z_t_i = np.squeeze(z_t_i)
        self.m = np.max(z_t_i, axis=1)  # max val for each article
        self.exp_sum = [sum(np.exp(z_t_i[t, i]-self.m[t]) if (z_t_i[t, i]-self.m[t]) >= -self.k else 0 
                            for i in range(self.topic_num)) for t in range(self.N)]
        self.w_t_i = np.array([[np.exp(z_t_i[t, i]-self.m[t]) / self.exp_sum[t] if (z_t_i[t, i]-self.m[t]) >= -self.k else 0
                               for i in range(self.topic_num)] for t in range(self.N)])  # row - t, colu - i

    def M_step(self):
        self.a_i = (1. / self.N) * np.sum(self.w_t_i, axis=0)  # for every topic we sum all the articles
        if sum(1 if a < self.epsilon else 0 for a in self.a_i) > 0:
            self.fix_a_i()
        for ix in range(self.topic_num):
            self.p_i_k[ix] = {word: ((sum(self.w_t_i[t, ix] * self.n_t_k[t].get(word, 0) for t in range(self.N)) + self.lmbda) /
                                     (sum(self.w_t_i[t, ix] * self.n_t[t] for t in range(self.N)) + self.lmbda * len(self.corpus_freq)))
                              for word in self.corpus_freq.keys()}

    def fix_a_i(self):
        self.a_i = [max(a, self.epsilon) for a in self.a_i]
        self.a_i = self.a_i / sum(self.a_i)

    def init(self):
        self.w_t_i = np.zeros([self.N, self.topic_num])
        for i, label in enumerate(self.labels):
            ordinal_number = int(label[0]) % self.topic_num
            self.w_t_i[i, ordinal_number] = 1.

        # for every article we have a list of word_freq in each topic -> # row - t, colu - i
        z_t_i = [[np.log(self.a_i[ix]) + sum(self.n_t_k[t][word] * np.log(freq)for word, freq in self.p_i_k[ix].items())
                  for ix in range(self.topic_num)] for t in range(self.N)]
        z_t_i = np.squeeze(z_t_i)
        self.m = np.max(z_t_i, axis=1)  # max val for each article
        self.exp_sum = [sum(np.exp(z_t[i] - self.m[i]) if (z_t[i] - self.m[i]) >= -self.k else 0 for i in range(self.topic_num))
                        for z_t in z_t_i]

    def loglikelihood(self):
        z_t_i = [[exp * w for w in w_t] for exp, w_t in zip(self.exp_sum, self. w_t_i)]
        log_ll = [m + sum(z_t) for m, z_t in zip(self.m, z_t_i)]
        log_ll = sum(log_ll)
        return log_ll

    def perplexity(self):
        res = self.loglikelihood()
        res = res / self.total_num_of_word
        res = np.exp(-res)
        return res

    def run(self, delta):
        log_ll = []
        perplexity = []
        while True:
            if len(log_ll) > 3 and abs(log_ll[-1] - log_ll[-3]) < delta:
              break
            self.M_step()
            log_ll.append(self.loglikelihood())
            perplexity.append(self.perplexity())
            self.E_step()
            print('loglikelihood - {}    perplexity - {}'.format(log_ll[-1], perplexity[-1]))
        plot_graph(log_ll[1:], 'loglikelihood', 1)
        plot_graph(perplexity[1:], 'Perplexity', 2)

    def filter_rare_words(self, corpus, rare_w_fltr):
        one_arr = [word for article in corpus for word in article]
        word_freq = Counter(one_arr)
        rarewords = Counter({word: freq for word, freq in word_freq.items() if freq <= rare_w_fltr})
        return word_freq - rarewords

    def read_article(self, f_name):
        text = [[word for word in line.split()] for i, line in enumerate(open(f_name, 'r').readlines()) if i % 4 == 2]
        self.corpus_freq = self.filter_rare_words(text, self.rare_w_th)
        self.articles_without_rare = [[word for word in article if word in self.corpus_freq.keys()] for article in text]
        self.labels = [line.rstrip(">\n").split()[1:] for i, line in enumerate(open(f_name, 'r').readlines())
                       if i % 4 == 0]


def plot_graph(x, title, id):
    plt.figure(id)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.plot(x)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Missing command line arguments, correct format:\ndev_set_file')
    main(sys.argv[1])
