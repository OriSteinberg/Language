import sys
import math
from collections import Counter, defaultdict

V = 300000
UNSEEN_WORD = 'unseen-word'


def main(dev_set_file, test_set_file, input_word, output_file):
    # Adding init outputs
    outputs = [dev_set_file, test_set_file, input_word, output_file, V, 1 / V]

    dev_data = read_file(dev_set_file)
    test_data = read_file(test_set_file)

    # Adding pre-processing outputs
    outputs.append(len(dev_data))

    # Adding the lidstone estimations outputs
    lidstone_model, l_param, lidstone_outputs = estimate_lidstone_model(input_word, dev_data)
    outputs.extend(lidstone_outputs)

    # Adding the held-out estimations outputs
    held_out_model, held_out_outputs = estimate_held_out_model(input_word, dev_data)
    outputs.extend(held_out_outputs)

    # Adding the test data length
    outputs.append(len(test_data))

    evaluation_outputs, output_table = evaluate_models(test_data, dev_data, lidstone_model, held_out_model, l_param)

    # Adding test data outputs
    outputs.extend(evaluation_outputs)

    write_outputs(output_file, outputs, output_table)


def read_file(f_name):
    return [word for i, line in enumerate(open(f_name, 'r').readlines()) if i % 4 == 2 for word in line.split()]


def write_outputs(f_name, outputs, outputs_table):
    lines.extend(['\t'.join(['#Output' + str(i + 1), str(output)]) for i, output in enumerate(outputs)])
    lines.append('#Output' + str(len(outputs) + 1))
    lines.extend(['\t'.join([k, ' '.join(v)]) for k, v in outputs_table.items()])
    open(f_name, 'w').writelines('\n'.join(lines))


def estimate_lidstone_model(input_word, dev_data):
    outputs = []
    split = round(0.9 * len(dev_data))
    train = dev_data[:split]
    validation = dev_data[split:]
    model = LidstoneModel(train)

    # Adding lidstone pre-processing outputs
    outputs.extend([len(validation), len(train), len(set(train)), model.get_freq(input_word)])

    mle_input_word = model.calc_mle(input_word)
    mle_unseen_word = model.calc_mle(UNSEEN_WORD)
    lidstone_input_word = model.calc_lidstone(input_word, 0.1)
    lidstone_unseen_word = model.calc_lidstone(UNSEEN_WORD, 0.1)

    # Adding MLE for input word and unseen word
    outputs.extend([mle_input_word, mle_unseen_word, lidstone_input_word, lidstone_unseen_word])

    # Adding perplexity values for lambda = 0.01, 0.1, 1
    outputs.extend([model.calc_perplexity(validation, l_param) for l_param in [0.01, 0.1, 1]])

    # Shuffle data and test this to find best l_param
    all_perplexities = {model.calc_perplexity(validation, l_param / 100): l_param / 100 for l_param in range(1, 200)}
    min_perplexity = min(all_perplexities.keys())
    min_l_param = all_perplexities[min_perplexity]

    # Adding the lambda value which gives minimal perplexity on the validation and the minimal perplexity
    outputs.extend([min_l_param, min_perplexity])
    return model, min_l_param, outputs


def estimate_held_out_model(input_word, dev_data):
    outputs = []
    split = round(0.5 * len(dev_data))
    train = dev_data[:split]
    held_out = dev_data[split:]
    model = HeldOutModel(train, held_out)

    # Add held-out pre-processing outputs
    outputs.extend([len(train), len(held_out)])

    held_out_input_word = model.calc_held_out(input_word)
    held_out_unseen_word = model.calc_held_out(UNSEEN_WORD)

    # Adding held-out estimations for the input and unseen word
    outputs.extend([held_out_input_word, held_out_unseen_word])

    return model, outputs


def evaluate_models(test_data, dev_data, lidstone_model, held_out_model, l_param):
    test_perplexity_lidstone = lidstone_model.calc_perplexity(dev_data, l_param)
    test_perplexity_held_out = held_out_model.calc_perplexity(dev_data)
    better_perplexity = 'L ' if (test_perplexity_lidstone < test_perplexity_held_out) else 'H'

    # Adding test data outputs
    outputs = [test_perplexity_lidstone, test_perplexity_held_out, better_perplexity]

    # Creating output table
    output_table = {str(i): (
        str(round(lidstone_model.calc_lidstone_by_freq(i, l_param) * len(dev_data), 5)),
        str(round(held_out_model.calc_held_out_by_freq(i) * len(dev_data), 5)),
        str(held_out_model.get_N_r(i)),
        str(held_out_model.get_t_r(i)))
                    for i in range(10)}

    # return test data outputs
    return outputs, output_table


class LidstoneModel:
    def __init__(self, train):
        self.train = train
        self.train_freq = Counter(train)

    def get_freq(self, word):
        return self.train_freq.get(word, 0)

    def calc_mle(self, word):
        return self.train_freq[word] / len(self.train)

    def calc_lidstone(self, word, l_param):
        return self.calc_lidstone_by_freq(self.get_freq(word), l_param)

    def calc_lidstone_by_freq(self, word_freq, l_param):
        return (word_freq + l_param) / (len(self.train) + l_param * V)

    def calc_perplexity(self, validation, l_param=0.06):
        sum_log_estimates = sum([math.log2(self.calc_lidstone(word, l_param)) for word in validation])
        return math.pow(2, -sum_log_estimates / len(validation))


class HeldOutModel:
    def __init__(self, train, held_out):
        self.train = train
        self.held_out = held_out
        self.train_freq = Counter(train)
        self.N_r_freq = Counter(self.train_freq.values())

        # Set N_0 to be the count of all the words that don't appear in the train
        self.N_r_freq[0] = V - len(self.train_freq)

        # Temp dicts to create t_r
        held_out_freq = Counter(held_out)
        train_freq_inv = defaultdict(list)
        for word, freq in self.train_freq.items():
            train_freq_inv[freq].append(word)
        train_freq_inv[0].extend(set(held_out) - set(train))

        self.t_r = {freq: sum(held_out_freq.get(word, 0) for word in words_with_freq)
                    for freq, words_with_freq in train_freq_inv.items()}

    def get_freq(self, word):
        return self.train_freq.get(word, 0)

    def calc_held_out(self, word):
        return self.calc_held_out_by_freq(self.get_freq(word))

    def calc_held_out_by_freq(self, word_freq):
        r = word_freq
        a = self.get_t_r(r)
        b = self.get_N_r(r)
        c = len(self.held_out)
        return (self.get_t_r(r) / self.get_N_r(r)) / len(self.held_out)

    def calc_perplexity(self, test_data):
        sum_log_estimates = sum([math.log2(self.calc_held_out(word)) for word in test_data])
        return math.pow(2, -sum_log_estimates / len(self.held_out))

    def get_N_r(self, r):
        return self.N_r_freq[r] if r != 0 else self.N_r_freq[r]

    def get_t_r(self, r):
        return self.t_r[r]


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise Exception('Missing command line arguments, correct format:\ndev_set_file, test_set_file, input_word, output_file')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
