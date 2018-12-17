#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# Shashank Shekhar(shashekh)
# Ishneet Arora(iarora)
# Bivas Maiti(bmaiti)
# (based on skeleton code by D. Crandall, Oct 2018)
#
# We are doing an Optical Character Recognition, based on the training images that have been supplied and a text file
# to learn the transition probabilities from one state to another.
#
# The image- breakdown to pixels
#
# Each character of the training set and test set have been broken down to lists of pixels, with the value * determining
# that the pixel is full, and space determining that it is empty. Each character is 14 pixel wide and 25 pixel in height,
# so we have a 2d array of 14 columns and 25 rows to represent each character.
#
# Some points about Noise:
#
# Our Assumption is that there is no noise in the training data character images. This means the characters should look exactly
# like the ones in the training data image. But, there is considerable noise in the test data which makes it
# difficult to accurately determine the character given its image, since not all the pixels match. Now we take help of
# our knowledge of the English language to increase our accuracy of our recognition system. For example if a system cant
# decide if a character is i or 1, we will consider the letter just before it to help it make that decision. i.e.
# We assume that each letter is dependent on the one before that. Now we have ourselves a hidden markov model.
#
# The Training Data Structure
# To get the transition probabilities, we are creating a dictionary data strucure. The Structure is
# as follows:
#  { transition: { from_char(70 in count): {to_char(70 in count): count_of_transition, _start: starting_instance_of_char,
# _count: Total_Instances_of_char, _end: Ending_Instances_of_char}},
#  { sentences_count : total_sentences_count}}
# We used several books from gutenberg.org to generate the Training Data Structure for the transition probabilities.
#
# The Models:
# Simplified Model:
# Here we consider only the train_image and test_image, not the language training. This can also be called the
# emission probability. We are calculating this probability as follows:
# we look pixel by pixel and check if there is a match between training and test images of the characters.
# We take product=1
# If we get a match
#   If the match is a filled pixel(i.e. *)
#       Multiply product with 0.8
#   Else
#       Multiply product with 0.6
# #Since if we get a filled-pixel match, likelihood of the characters being the same is high, on the other hand, if its
# an empty pixel match, that likelyhood goes down, since a lot of pixels are empty in a lot of characters.
# If we dont get a match
#   If train_letter pixel is filled(i.e. *) and test_letter pixel is empty (i.e. space)
#       Multiply product with 0.4
#   Else
#       Multiply product with 0.2
# This is assuming that a lot of pixels in the test images are not filled due to the noise, so we are still multiplying 0.4
# to the product even if we get a white space instead of a filled pixel. But when we are getting a filled pixel in the
# test letter pixel when it is empty in the train letter pixel, it is very likely to be a different character than the
# training one
# We are assuming that each letter/Character depends only on the last character that was present. This means we have
# ourselves a hidden markov model, where we are trying to guess a sequence of characters(unobserved variables) given
# some observed variables(the image of the string).
#
# HMM Model:
# Since we have an HMM at our hands we can find the sequence of most probable characters using the Viterbi Algorithm.
# Below is the Viterbi Algorithm:
#
# V[t+1][char]=E[train_char][test_char] * MAX over all prev_chars {V[t][prev_char]* P[prev_char][char]}
#
# where,
# V[t+1][char]= Probability that system has state = char at time t+1
# E[train_char][test_char]=Probability of observing the test_char given that it is in state = train_char at time t+1
# P[prev_char][char]= Probability of the system transitioning from state prev_char to char
# V[t][prev_char]=Probability that system has state = prev_char at time t
#
# Among these values,
# E[tag][word] needs to be calculated. It is the same as the emission probability in the simplified model.
#
# Other values in the implementation of Viterbi include:
# P[prev_char][char]-> can be got from the transition part of the DS.
# We go on filling the values of V[t][char] for all the chars for all times starting from t=0 to the length of the
# string. We keep on storing these values so we can get their values for the next values of t
# In the end we are present with a table for V[t][char] for all values of t and chars.
# We also keep track of the prev_char value which produces the MAX of V[t][prev_char]* P[prev_char][char], and store it in
# a list of list of dictionaries.
# Each dictionary entry will look like - char: prev_char
# Each t value will have a list of dictionaries.
# Finally we'll have a list of list of dictionaries.
#
# Now, since we have the maximum value of V[t+1][char] where t+1 is the length of the string, we'll consider the most
# probable char for t+1 to be the one where the V value is maximized. Now, using our list of dictionaries, we will back
# track to find the most likely sequence of chars.

import math

from PIL import Image, ImageDraw, ImageFont
from text_trainer import Trainer_builder
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
text_trainer=Trainer_builder(train_txt_fname)
def solve_simple():
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    simple_output=[]
    for character in test_letters:
        max_prob = float('-inf')
        max_char = ''
        for train_char in TRAIN_LETTERS:
            emission=get_emission(train_letters[train_char],character)
            if  emission> max_prob:
                max_prob=emission
                max_char=train_char
        simple_output.append(max_char)
    return "".join(simple_output)

def get_emission(train_letter,test_letter):
    product=1.0
    for i in range(0,CHARACTER_HEIGHT):
        for j in range(0,CHARACTER_WIDTH):
            if train_letter[i][j]==test_letter[i][j]:
                if train_letter[i][j]=="*":
                    product*=0.9
                else:
                    product *= 0.6
            else:
                if train_letter[i][j]=="*":
                    product*=0.4
                else:
                    product *= 0.1
    return math.log(product)

def viterbi_solve():
    training_data=text_trainer.trainer
    letter_tracker=[]
    V=[]
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for test_letter in test_letters:
        v_states = {}
        letter_tracker_max = {}
        for letter in TRAIN_LETTERS:
            emission=get_emission(train_letters[letter],test_letter)
            max_last_prob_transition = float('-inf')
            max_char = ''
            if len(V)>0:
                for prev_letter in TRAIN_LETTERS:
                    if (training_data['transition'][prev_letter][letter] > 0):
                        last_state_transition = V[len(V) - 1][prev_letter] + math.log(float(training_data['transition'][prev_letter][letter]) / float(training_data['transition'][letter]['_count']))
                    else:
                        if training_data['transition'][letter]['_count']>0:
                            last_state_transition = V[len(V) - 1][prev_letter] + math.log(float(.00000000001) / float(training_data['transition'][letter]['_count'] ))
                        else:
                            last_state_transition= V[len(V) - 1][prev_letter]+math.log(float(.0000000000000001))
                    if (max_last_prob_transition < last_state_transition):
                        max_last_prob_transition = last_state_transition
                        max_char = prev_letter
                v_states[letter] = emission + max_last_prob_transition
                letter_tracker_max[letter] = (max_char)
            else:
                if training_data['transition'].get(letter, False) == False:
                    initial_prob=math.log(float(.000001))
                elif (training_data['transition'][letter]['_first'] == 0):
                    initial_prob = math.log(float(.000001))
                else:
                    initial_prob = math.log(float(training_data['transition'][letter]['_first']) / float(training_data['sentences_count']))
                v_states[letter] = initial_prob + emission
        V.append(v_states)
        letter_tracker.append(letter_tracker_max)
    return "".join(get_sequence(V,letter_tracker))

def get_sequence(V,letter_tracker):
    sequence=[]
    max_prob_tag = max(V[len(test_letters)-1].items(), key=lambda item: item[1])[0]
    sequence.append(max_prob_tag)

    for i in range(len(test_letters)-1,0,-1):
        max_prob_tag=letter_tracker[i][max_prob_tag]
        sequence.append(max_prob_tag)
    return sequence[::-1]

# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,

print "Simple:   "+ solve_simple()
viterbi_output=viterbi_solve()
print "Viterbi:  "+viterbi_output
print "Final Answer:\n"+viterbi_output


