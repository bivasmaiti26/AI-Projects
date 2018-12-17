###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Shashank Shekhar(shashekh)
# Ishneet Arora(iarora)
# Bivas Maiti(bmaiti)
# (Based on skeleton code by D. Crandall)
#
#
####
# There are 3 models for which parts of speech tagging(POS) has been done here. These are Simplified, HMM and Complex.
# The POS can be thought of as unobserved variable. And the word can be considered as the observed variable.
# We will be using the Notations S[i] for the POS tag of the i th word of the sentence, w[i] for the actual word itself.
#
# The Training Data Structure
# To get the prior, transition and emission probabilities, we are creating a dictionary data strucure. The Structure is
# as follows:
#  { transition: { from_tag(12 in count): {to_tag(12 in count): count_of_transition,_start: starting_instance_of_tag,
# _count: Total_Instances_of_Tag, _end: Ending_Instances_of_Tag}},
# { double_transition: { from_tag1_then_tag2(144 in count): {to_tag(12 in count): count_of_transition,_start: starting_instance_of_tag1_tag2,
# # _count: Total_Instances_of_Tag1_tag2, _end: Ending_Instances_of_Tag1_tag2}},
#
# { sentences_count : total_sentences_count},
#  { word_count: total_word_count_},
#  {emission: word[i]: {tag(12 count in total):total_tags_of_word_count, _count: total_word_count of word[i] }} }
#
# The Models:
#
# Simplified model:
# Word Accuracy:93.94% for 2000 sentences in bc.test
# Sentence Accuracy: 47.55% for 2000 sentences in bc.test
# This is where a word is only dependent on the corresponding POS, and any POS tag is not dependent on other
# tags given the word(observable).
# So we need to find:s[i]= arg max on all si belongs to Si * P(Si = si|W)
# To get Output tags: We find:
# For each word in sentence,
#       For each tag
#           Find the value P(S[i]|w[i]).
#       Find the tag which maximizes the above value for that word.
# Return all the tags with the list of tags which maximizes the value of P(Si = si|W)
#
# To find the posterior of a sequence of tags using the Simplified Model, below is the equation:
# product over all values of i {P(S[i]) | P(w[i]) },
# where i ranges from 0 to the length of the sentence.
#
#
# Hidden Markov Model(HMM):
# Word Accuracy: 94.59% for 2000 sentences in bc.test
# Sentence Accuracy: 49.00 for 2000 sentences in bc.test
#
# In this model, we now add another layer of dependency:
# We assume that each POS tag depends on its previous tag. Now we have a Hidden Markov Chain,
# where observed variables are the words, and unobserved variables are the corresponding POS tags. Each word is
# dependent on the POS tag, and each POS tag is dependent on the previous POS tag.
# Since this is an HMM, we can use Viterbi Algorithm to get the most probable tags:
# Below is the Viterbi Algorithm:
# V[t+1][tag]=E[tag][word] * MAX over all prev_tags {V[t][prev_tag]* P[prev_tag][tag]}
# where,
# V[t+1][tag]= Probability that system has state = tag at time t+1
# E[tag][word]=Probability of observing the word given that it is in state = tag at time t+1
# P[prev_tag][tag]= Probability of the system transitioning from state prev_tag to tag
# V[t][prev_tag]=Probability that system has state = prev_tag at time t
#
# Among these values,
# E[tag][word] can be got from the emission part of the DS.
# P[prev_tag][tag] can be got from the transition part of the DS.
# We go on filling the values of V[t][tag] for all the tags for all times starting from t=0 to the length of the
# sentence. We keep on storing these values so we can get their values for the next values of t
# In the end we are present with a table for V[t][tag] for all values of t and tags.
# We also keep track of the prev_tag value which produces the MAX of V[t][prev_tag]* P[prev_tag][tag], and store it in
# a list of list of dictionaries.
# Each dictionary entry will look like - tag: prev_tag
# Each t value will have a list of dictionaries.
# Finally we'll have a list of list of dictionaries.
#
# Now, since we have the maximum value of V[t+1][tag] where t+1 is the length of the sentence, we'll consider the most
# probable tag for t+1 to be the one where the V value is maximized. Now, using our list of dictionaries, we will back
# track to find the most likely sequence of tags.
#
# To find the posterior of a sequence of tags using the HMM Model, below is the equation:
#
# P(S[1]) * P(w[1]|S[1]) product over all values of t=2 to length of the sentence(P(S[t] | S[t-1] ) * P(w[t] | S[t] ))
#
#
# The Complex Model.
# Word Accuracy: 93.46% for 2000 sentences in bc.test
# Sentence Accuracy: varies from 44% to 47% for 2000 sentences in bc.test
#
# Here along with the HMM, which says each tag is dependent on its previous tag, we add one more layer of # depepndency to get the complex mode. We assume that each tag is dependent on its previous tag and the one before that # as well. Now this model is no longer a Hidden Markov Chain, and needs to be approached in a different way. We are # using Markov Chain Monte Carlo(MCMC), a sampling method to get the most probable tags. We implement an algorithm called # Gibbs' Algorithm to generate a number of samples and then get the most probable tags from that sample.
#
# The Gibbs Sampler:
# The main idea is to sample particles through large iterations based on a complicated bayes net (which would be a computational overhead if we solve using variable elimination).
# After many iterations, we observe that the  posterior  is very close to the problem we are going to solve.
# The number of iterations after which the posterior value comes close to the posterior we want to find out is called burn in period.
#
# Algorithm:
#       1. Generate one random sample, x[0]
#       2. For each sample s = 1 .. S
#          3. Copy the previous sample, x[s] = x[s-1]
#          4. For each unobserved variable Xi,
#             5. Sample the value of Xi given values for all other variables in X[s]
#             6. Put this sampled value in x[s]
#
# Step 5 of the algorithm is crucial because we calculate the distribution of Xi variable using factorization of given bayes net model; # P(S1, S2, S3, .. Sn) = P(S1)P(S2|S1)P(S3|S2,S1) ... P(Sn|Sn-2, Sn-1 )P(O1|S1)P(O2|S2)P(O3|S3)...P(On|Sn).
# So, for example, if we are to find the distribution of S2, we consider all terms which touch S2; P(S2) = P(S2|S1)P(S3|S2,S1)P(S4|S3,S2)P(O2|S2)
# This way, we get the distribution, now we use np.random.choice function as a biased coin flip to pick a value from this distribution.
#
# After a burn in of 200 iterations, we observed that we were very near to the posterior we desired to calculate.
#
# Calculation of probability distribution of unobserved variables:
#
# There will be five cases: if fixed variable is the first one of the sentence, second one, last one, last but one, or
# anything else.
# Below are the priors of the five cases
#
# Case 1: index of word=1
# P(S1|S2,S3,w1)=P(S1)P(w1|S1)P(S2|S1)P(S3|S2,S1)
#
# Case 2: index of word=2
# P(S2|S1,S3,S4,w2)=P(S1)P(S2|S1)P(S3|S2,S1)P(S4|S3,S2)P(w2|S2)
#
# Case 3: index of word=last
# P(Sn|Sn-1,Sn-2,wn)=P(Sn-2)P(Sn-1|Sn-2)P(Sn|Sn-1,Sn-2)P(wn|Sn)
#
# Case 4: index of word=last but one
# P(Sn-1|Sn-3,Sn-2,Sn,wn-1)=P(Sn-3)P(Sn-2|Sn-3)P(Sn-1|Sn-3,Sn-2)P(Sn|Sn-1,Sn-2)P(wn-1|Sn-1)
#
# Case 5: anything else except the above 4 cases
# P(Si|Si-1,Si-2,Si+1,Si+2,wi)=P(Si-2)P(Si-1|Si-2)P(Si|Si-1,Si-2)P(Si+2|Si+1,Si)P(Si+1|Si)P(wi|Si)
#
#
# Calculation of Posterior:
# P(S1)P(S2|S1)P(S3|S2,S1)....P(Sn|Sn-1,Sn-2)P(w1|S1)P(w2|s2)P(w3|S3).....P(wn|Sn)
#
#
# #
####

import random
import math
from data_loader import Trainer_builder
from gibbs import GibbsSampler
from hmm_solver import Viterbi
from simple_solver import Simplified

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!

    def __init__(self,train_file_name, test_file_name):
        self.train_file=train_file_name
        #self.trainer_builder = Trainer_builder(self.train_file).get_trainer()
        #a = GibbsSampler(self.trainer_builder)
        #label =  a.do_gibbs('The discovery struck Nick like a blow .'.split())
        # a.calculate_posterior(('at', 'the', 'same', 'instant', ',', 'nick', 'hit', 'the', 'barrel', 'and', 'threw', 'himself', 'upon', 'the', 'smaller', 'man', '.'), map(lambda item: item.upper(), label))

    def posterior(self, model, sentence, label):
        if model == "Simple":
            simple_instance = Simplified(sentence)
            return simple_instance.calc_posterior(self.trainer_builder, label)
        elif model == "Complex":
            return self.gibbs_instance.calculate_posterior(sentence, map(lambda item: item.upper(), label))
        elif model == "HMM":
            viterbi_instance = Viterbi(sentence)
            return viterbi_instance.calc_posterior(self.trainer_builder,label)
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        self.trainer_builder = Trainer_builder(self.train_file).get_trainer()

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        simple_instance = Simplified(sentence)
        return simple_instance.get_most_probable_tags(self.trainer_builder)

    def complex_mcmc(self, sentence):
        self.gibbs_instance = GibbsSampler(self.trainer_builder)
        return self.gibbs_instance.do_gibbs(sentence)

    def hmm_viterbi(self, sentence):
        viterbi_instance= Viterbi(sentence)
        self.hmm_tags= viterbi_instance.get_most_probable_tags(self.trainer_builder)[0]
        return self.hmm_tags

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            output= self.hmm_viterbi(sentence)
            return output
        else:
            print("Unknown algo!")

