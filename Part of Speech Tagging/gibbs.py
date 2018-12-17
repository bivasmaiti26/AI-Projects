import random

import numpy as np
import math


class GibbsSampler:
    def __init__(self, trainer):
        self.trainer = trainer
        self.tags = ['ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']
        self.iterations = 1000
        self.min_probability = pow(10, -20)
        self.burn_in = 200

    # returns a biased random item of a given
    # 'sequence' if provided a
    # 'distribution' as a list of probabilities.
    # The sum of all items in the
    # distribution should be 1.
    def random_on_distribution(self, sequence, distribution=None):
        return np.random.choice(sequence, 1, p=distribution)[0]

    # returns a random number less than 'max'.
    # pass the length of sentence to get a random position.
    def random_position(self, max):
        return np.random.randint(0, max, size=1)[0]

    # implementation of gibbs sampling.
    def do_gibbs(self, sentence):
        len_of_sentence = len(sentence)
        # an empty 2D array to hold samples per iteration.
        X = [['' for j in range(len_of_sentence)] for i in range(self.iterations)]
        pos_frequency = [{tag: 0 for tag in self.tags} for i in range(len_of_sentence)]
        # random sample.
        X[0] = [self.random_on_distribution(self.tags) for i in range(len_of_sentence)]
        for i in range(1, self.iterations):
            # copy from previous sample.
            X[i] = list(X[i - 1])
            # get a random index
            random_index = self.random_position(len_of_sentence)
            # get distribution of this unobserved variable, given all other variables.
            distribution_ds = self.dict_to_distribution_list(self.distribution_given_other_variables(random_index, X[i], sentence))
            # biased coin flip and select one value of random variable at index random_index
            X[i][random_index] = self.random_on_distribution(distribution_ds['tags'], distribution_ds['distribution'])
            # wait for burn in, then store the tags
            if i >= self.burn_in:
                for it, tag0 in enumerate(X[i]):
                    pos_frequency[it][tag0] += 1
        # get most probable tags and convert into lower case
        sequence = map(lambda item: item.lower() if item != 'X' else 'X', self.convert_into_tags(pos_frequency))
        return sequence

    def convert_into_tags(self, pos_frequency):
        tags = []
        for tags_frequncy in pos_frequency:
            tags.append(max(tags_frequncy.items(), key=lambda item: item[1])[0])
        return tags

    # P(S[i])
    def prob_initial_or_prior(self, index, tag):
        MIN_NUMBER = self.min_probability
        product = 1
        if index == 0:
            if self.trainer['transition'][tag]['_start'] > 0:
                product *= float(self.trainer['transition'][tag]['_start']) / float(self.trainer['sentences_count'])
            else:
                product *= MIN_NUMBER
        else:
            product *= float(
                self.trainer['transition'][tag]['_count'] + self.trainer['transition'][tag]['_end']) / float(self.trainer['word_count'])
        return product

    # P(S[i]|S[i-1])
    def prob_state_given_previous(self, prev_tag, tag):
        product = 1
        MIN_NUMBER = self.min_probability
        if self.trainer['transition'][prev_tag][tag] > 0:
            product *= float(self.trainer['transition'][prev_tag][tag]) / float(self.trainer['transition'][prev_tag]['_count'])
        else:
            product *= MIN_NUMBER
        return product

    # P(S[i]|S[i-1],S[i-2])
    def prob_state_given_prev_two_states(self, last_tag, prev_tag, tag):
        product = 1
        MIN_NUMBER = self.min_probability
        if last_tag + '_' + prev_tag in self.trainer['double_transition']:
            if tag in self.trainer['double_transition'][last_tag + '_' + prev_tag]:
                if self.trainer['double_transition'][last_tag + '_' + prev_tag]['_count'] > 0 and self.trainer['double_transition'][last_tag + '_' + prev_tag][tag] > 0:
                    product *= float(self.trainer['double_transition'][last_tag + '_' + prev_tag][tag]) / float(self.trainer['double_transition'][last_tag + '_' + prev_tag]['_count'])
                else:
                    product *= MIN_NUMBER
            else:
                product *= MIN_NUMBER
        else:
            product *= MIN_NUMBER
        return product

    # P(w[i]|S[i])
    def prob_state_given_word(self, word, tag):
        product = 1
        MIN_NUMBER = self.min_probability
        if word in self.trainer['emission'] and self.trainer['emission'][word][tag] > 0:
            product *= float(self.trainer['emission'][word][tag]) / float(self.trainer['transition'][tag]['_count'] + self.trainer['transition'][tag]['_end'])
        else:
            product *= MIN_NUMBER if tag != "NOUN" or word in self.trainer['emission'] else 1
        return product

    def distribution_given_other_variables(self, index, sample_tags, sentence):
        MIN_NUMBER = self.min_probability
        prob_dist = {}
        marginalized_sum = 0
        # Fixing S1
        if index == 0:
            for tag in self.tags:
                prob_product = 1
                # P(S1)-Take Initial
                prob_product *= self.prob_initial_or_prior(index, tag)
                # P(w1|S1)
                prob_product *= self.prob_state_given_word(sentence[index], tag)
                # P(S2|S1)
                prob_product *= self.prob_state_given_previous(tag, sample_tags[1]) if len(sentence) > 1 else 1
                # P(S3|S2,S1)
                prob_product *= self.prob_state_given_prev_two_states(tag, sample_tags[1], sample_tags[2]) if len(
                    sentence) > 2 else 1
                marginalized_sum += prob_product
                prob_dist[tag] = prob_product
        # Fixing S2
        elif index == 1:
            for tag in self.tags:
                prob_product = 1
                # P(S1)-Take Initial
                prob_product *= self.prob_initial_or_prior(index - 1, tag)
                # P(w2|S2)
                prob_product *= self.prob_state_given_word(sentence[index], tag)
                # P(S2|S1)
                prob_product *= self.prob_state_given_previous(sample_tags[0], tag)
                # P(S3|S2,S1)
                prob_product *= self.prob_state_given_prev_two_states(sample_tags[0], tag, sample_tags[2]) if len(
                    sentence) > 2 else 1
                # P(S4|S3,S2)
                prob_product *= self.prob_state_given_prev_two_states(tag, sample_tags[2], sample_tags[3]) if len(
                    sentence) > 3 else 1
                marginalized_sum += prob_product
                prob_dist[tag] = prob_product
        # Fixing Sn
        elif index == len(sentence) - 1:
            last_but_one_tag = sample_tags[index - 2]
            last_but_two_tag = sample_tags[index - 3]
            for tag in self.tags:
                prob_product = 1
                # P(wn|Sn)
                prob_product *= self.prob_state_given_word(sentence[index], tag)
                # P(Sn|Sn-1,Sn-2)
                prob_product *= self.prob_state_given_prev_two_states(last_but_two_tag, last_but_one_tag, tag)
                # P(Sn-1|Sn-2)
                prob_product *= self.prob_state_given_previous(last_but_two_tag, last_but_one_tag)
                # P(Sn-2)-Take Prior
                prob_product *= self.prob_initial_or_prior(index - 2, last_but_two_tag)

                marginalized_sum += prob_product
                prob_dist[tag] = prob_product
        # Fixing Sn-1
        elif index == len(sentence) - 2:
            last_tag = sample_tags[index - 1]
            last_but_two_tag = sample_tags[index - 3]
            last_but_three_tag = sample_tags[index - 4]
            for tag in self.tags:
                prob_product = 1
                # P(wn-1|Sn-1)
                prob_product *= self.prob_state_given_word(sentence[index], tag)
                # P(Sn-1|Sn-2,Sn-3)
                prob_product *= self.prob_state_given_prev_two_states(last_but_three_tag, last_but_two_tag, tag)
                # P(Sn|Sn-1,Sn-2)
                prob_product *= self.prob_state_given_prev_two_states(last_but_two_tag, tag, last_tag)
                # P(Sn-2|Sn-3)
                prob_product *= self.prob_state_given_previous(last_but_three_tag, last_but_two_tag)
                # P(Sn-3)- Take Prior
                prob_product *= self.prob_initial_or_prior(index - 2, last_but_three_tag)
                marginalized_sum += prob_product
                prob_dist[tag] = prob_product
        # Fixing Si
        else:
            prev_tag = sample_tags[index - 1]
            prev_second_tag = sample_tags[index - 2]
            next_tag = sample_tags[index + 1]
            next_second_tag = sample_tags[index + 2]
            for tag in self.tags:
                prob_product = 1
                # P(Wi|Si)
                prob_product *= self.prob_state_given_word(sentence[index], tag)
                # P(Si|Si-1,Si-2)
                prob_product *= self.prob_state_given_prev_two_states(prev_second_tag, prev_tag, tag)
                # P(Si+2|Si, Si+1)
                prob_product *= self.prob_state_given_prev_two_states(tag, next_tag, next_second_tag)
                # P(Si-1|Si-2)
                prob_product *= self.prob_state_given_previous(prev_second_tag, prev_tag)
                # P(Si+1|Si)
                prob_product *= self.prob_state_given_previous(tag, next_tag)
                # P(Si-2)->Take Prior if i-2!=0
                prob_product *= self.prob_initial_or_prior(index - 2, prev_second_tag)
                marginalized_sum += prob_product
                prob_dist[tag] = prob_product
        # marginalized_sum=sum()
        for prob in prob_dist:
            prob_dist[prob] = prob_dist[prob] / marginalized_sum
        return prob_dist

    # since numpy np.random.choice needs a sequence and its distribution,
    # we need to convert the distribution dictionary into sequence list
    # and probability list.
    # for example, consider the distribution dictionary {'Noun': 0.5, 'ADJ': 0.6, 'PRT': 0.7, ...}
    # the return value will be of the form
    # {'tags': ['Noun', 'ADJ', 'PRT', ...], 'distribution': [0.5, 0.6, 0.7, ...]}
    def dict_to_distribution_list(self, distribution_dict):
        distribution_list_ds = {'tags': [], 'distribution': []}
        for tag, probability in distribution_dict.items():
            distribution_list_ds['tags'].append(tag)
            distribution_list_ds['distribution'].append(probability)
        return distribution_list_ds

    def calculate_posterior(self, sentence, label):
        MIN_NUMBER = self.min_probability
        log_sum_posterior = 0
        P_s1 = float(self.trainer['transition'][label[0]]['_start']) / self.trainer['sentences_count']
        if P_s1 == 0:
            P_s1 = MIN_NUMBER
        log_sum_posterior += math.log(P_s1)
        P_s2_given_s1 = float(self.trainer['transition'][label[0]][label[1]]) / self.trainer['transition'][label[0]]['_count'] if len(sentence)>1 else 1
        if P_s2_given_s1 == 0:
            P_s2_given_s1 = MIN_NUMBER
        log_sum_posterior += math.log(P_s2_given_s1)
        for i in range(2, len(sentence)):
            if label[i - 2] + "_" + label[i - 1] in self.trainer['double_transition']:
                if label[i] in self.trainer['double_transition'][label[i - 2] + "_" + label[i - 1]]:
                    double_trans_posterior = self.trainer['double_transition'][label[i - 2] + "_" + label[i - 1]][label[i]] / self.trainer['double_transition'][label[i - 2] + "_" + label[i - 1]][
                        '_count']
                    if double_trans_posterior == 0:
                        double_trans_posterior = MIN_NUMBER
                else:
                    double_trans_posterior = MIN_NUMBER
            else:
                double_trans_posterior = MIN_NUMBER
            log_sum_posterior += math.log(double_trans_posterior)

        for i in range(len(sentence)):
            if self.trainer['emission'].get(sentence[i], False) != False and label[i] in self.trainer['emission'][sentence[i]]:
                P_word_given_pos = self.trainer['emission'][sentence[i]][label[i]] / self.trainer['transition'][label[i]]['_count']+self.trainer['transition'][label[i]]['_end']
                if P_word_given_pos == 0:
                    P_word_given_pos = MIN_NUMBER
            else:
                P_word_given_pos = MIN_NUMBER
            log_sum_posterior += math.log(P_word_given_pos)
        return log_sum_posterior
