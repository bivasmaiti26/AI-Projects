#!/bin/python



##############################
#####PLEASE READ README.MD####
##############################


import json
import re
import math

def fill_bag_of_words() :
    for location in TRAINING_DATA['details']:
        for word in TRAINING_DATA['details'][location]['words'] :
            if BAG_OF_WORDS.get(word,False) == False :
                BAG_OF_WORDS[word] = 1
            else :
                BAG_OF_WORDS[word] +=1

def read_test(filename) :
    with open(filename, "r") as f:
       for line in f:

           # regex to prevent new line character appended at the end of the word
           match = re.match(r'(.+) *\n', line)
           if match:
               line = match.group(1)

           actual_location = re.match(r'\w+,\w+', line).group()
           tweet = re.sub(r'\w+,\w+ ', '', line)

           # remove special characters
           # [!@#%^&*\-\+\.,:_"\(\)'?]
           tweet_cleaned = re.sub(r'[!@#%^&*\-\+\.,:_"\(\)\'?]', ' ', tweet)

           # remove the & and
           tweet_cleaned = re.sub(r' the | and ', ' ', tweet_cleaned)
           tweet_words = tweet_cleaned.split()

           # changing to lower case
           tweet_words = [word.lower() for word in tweet_words]
           tweet_cleaned = tweet_cleaned.lower()
           TEST_DATA.append({'actual_location' : actual_location, 'tweet_words' : tweet_words, 'tweet' : tweet, 'tweet_cleaned' : tweet_cleaned})

# builds the training data nto a DS that can be queried  efficiently.
# use dump_trainer() to dump the Training data to a file called 'trainer' for debugging
def build_trainer(fileName):
    with open(fileName, "r") as f:
        for line in f:
            # regex to prevent new line character appended at the end of the word
            match = re.match(r'(.+) *\n', line)
            if match:
                line = match.group(1)

            # take location as word before first space
            location = re.match(r'\w+,\w+',line).group()
            # remove location from line
            line = re.sub(r'\w+,\w+ ','',line)

            # remove special characters
            # [!@#%^&*\-\+\.,:_"\(\)'?]
            line = re.sub(r'[!@#%^&*\-\+\.,:_"\(\)\'?]',' ',line)

            # remove the & and
            line = re.sub(r' the | and ', ' ', line)

            tokens = line.split()
            words = tokens[1:]

            #changing to lower case
            words = [word.lower() for word in words]
            words_count = len(words)
            TRAINING_DATA['all_word_count'] += words_count
            TRAINING_DATA['all_location_count'] += 1

            # if the location is not present in the dictionary.
            # takes O(1)
            if TRAINING_DATA['details'].get(location, False) == False:
                TRAINING_DATA['all_location'].append(location)
                TRAINING_DATA['details'][location] = {'words': words, 'words_count': words_count, 'tweet_count': 1, 'location_probability' : 1}

            # else just update the dictionary with new  values.
            else:
                TRAINING_DATA['details'][location]['words'] += words
                TRAINING_DATA['details'][location]['words_count'] += words_count
                TRAINING_DATA['details'][location]['tweet_count'] += 1
                TRAINING_DATA['details'][location]['location_probability'] = float(TRAINING_DATA['details'][location]['tweet_count'] * (1.0) / TRAINING_DATA['all_location_count'])

# dumps all the data structures to files
# to help naive humans to understand whats goin on!
def dump_data(type):
    if type == 'trainer':
        with open('trainer', 'w') as file:
            file.write(json.dumps(TRAINING_DATA, sort_keys=False, indent=4))
    elif type == 'test':
        with open('test', 'w') as file:
            file.write(json.dumps(TEST_DATA, sort_keys=False, indent=4))
    elif type == 'output' :
        with open('output', 'w') as file:
            file.write(json.dumps(CLASSIFIED_DATA, sort_keys=False, indent=4))
    else :
        with open('maxoutput', 'w') as file:
            file.write(json.dumps(MAX_OUTPUT, sort_keys=False, indent=4))

# P(L=l)
# probability of occurance of a tweet of a location
def get_location_probability(location) :
    return TRAINING_DATA['details'][location]['location_probability']


# P(w | L=l)
# probabilty of occurance of a word in a given location
def get_probability_word_given_location(given_word, location) :
    word_list = TRAINING_DATA['details'][location]['words']
    count_given_word = word_list.count(given_word)
    p = count_given_word*(1.0)/TRAINING_DATA['details'][location]['words_count']
    return  ADJUSTMENT if p==0 else p

# P(L=l | w1, w2, w3, .... wn)
# probability of location given all tweeted words
def get_probability_location_given_tweet(tweet, words, location) :
    adj = adjustments(tweet, location)
    if adj != -1 :
        #print tweet, location
        return 1
    product = 1
    location_probability = get_location_probability(location)
    for word in words :
        word_probability = get_probability_word_given_location(word,location)
        product = product * word_probability
    return location_probability * product




# max( P(L=li | w1, w2, w3, .... wn) )
# actual classification on Naive Bayes algorithm
# returns maximum probability that given a tweet, location is L
def get_probability_of_all_loc_given_tweet(tweet_words,tweet) :
     return max([(get_probability_location_given_tweet(tweet, tweet_words,location),location) for location in TRAINING_DATA['all_location']], key=lambda item: item[0])[1]



# max 5 P(L=li | wi)
# calculates top 5 words of each location
def probability_of_each_loation_per_word() :
    output = []
    for location in TRAINING_DATA['all_location'] :
        output_map = {'location': location}
        word_output_list = []
        for word in BAG_OF_WORDS :
            word_output_list.append((word,get_probability_word_given_location(word, location) * TRAINING_DATA['details'][location]['words_count'] * 1.0 / TRAINING_DATA['all_word_count'] \
            / BAG_OF_WORDS[word] * 1.0 / TRAINING_DATA['all_word_count']))
        output_map['probabilities'] = sorted(word_output_list, key = lambda item : item[1], reverse = True)[0:5]
        output.append(output_map)
    return output





# ugly adjustments which sends a high probability if any of the words after # or @ symbol
# is a part of location string
def adjustments(tweet_cleaned, location) :
   if location == 'Los_Angeles,_CA' :
       for particular in [' la ', 'losangeles'] :
           if particular in tweet_cleaned :
               return 1
   if location == 'San_Francisco,_CA':
       for particular in ['san francisco', 'sanfrancisco', 'francisco']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Manhattan,_NY':
       for particular in [' ny ', 'manhattan', 'new york']:
           if particular in tweet_cleaned:
               return 1
   if location == 'San_Diego,_CA':
       for particular in ['sandiego', 'san diego']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Houston,_TX':
       for particular in ['houston', ' tx ']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Chicago,_IL':
       for particular in ['chicago', 'illinois']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Philadelphia,_PA':
       for particular in ['philadelphia', ' pa ']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Toronto,_Ontario':
       for particular in ['toronto', 'ontario']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Atlanta,_GA':
       for particular in ['atlanta', 'georgia', ' ga ']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Washington,_DC':
       for particular in [' dc ', 'washington' , 'district of columbia']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Boston,_MA':
       for particular in [' ma ', 'boston']:
           if particular in tweet_cleaned:
               return 1
   if location == 'Orlando,_FL':
       for particular in [' fl ', 'orlando', 'florida']:
           if particular in tweet_cleaned:
               return 1
   return -1


# classifies all tweets in the test data w.r.t training data.
# calculates accuracy on the go, to avoid many loops later.
def classify() :
    output = {'accuracy': 0.0, 'contents' : []}
    record_count = 0
    correct = 0
    for test in TEST_DATA :
        record_count += 1
        label = get_probability_of_all_loc_given_tweet(test['tweet_words'],test['tweet_cleaned'])
        actual_location = test['actual_location']
        tweet = test['tweet']
        if actual_location == label :
            correct += 1
        output['contents'].append((actual_location, label, tweet))
        output['accuracy'] = correct * 100.0 / record_count
    return output





# training data structure that gives probability values in O(1) time
TRAINING_DATA = {'all_location_count': 0, 'details': {}, 'all_location' :[], 'all_word_count' : 0}

# Data structure that will contain classified values
CLASSIFIED_DATA = None

# Data structure that will contain test data
TEST_DATA = []

# Data structure to put all the words and their frequency
BAG_OF_WORDS = {}

# when a word does not occur in a given location then take this value
ADJUSTMENT = math.pow(10,-9)

# Data structure that will store max 5 words of every location
MAX_OUTPUT = None


read_test('tweets.test1.clean.txt')
build_trainer('tweets.train.clean.txt')
fill_bag_of_words()


# run Naive Bayesian classification
CLASSIFIED_DATA = classify()


# write classified data in file
with open('classified_data.txt', 'w') as file:
    for contents in CLASSIFIED_DATA['contents']:
        file.write(" ".join(contents) + "\n")

print "Classification Done, output file geerated : classified_data.txt"
print "Accuracy : "+str(CLASSIFIED_DATA['accuracy'])
print ""
print "Now calculating top 5 words in all the locations..."
print ""

# calculating the top 5 words in each location
MAX_OUTPUT = probability_of_each_loation_per_word()
for content_map in MAX_OUTPUT :
    msg = content_map['location'] + " "
    for (word,probability) in content_map['probabilities'] :
        msg += word + " "
    print msg
