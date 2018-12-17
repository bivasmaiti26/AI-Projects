# Naive Bayesian Classifier for tweets

![!accuracy](accuracy.png)

#### Calculates below Probability :
Probability that a tweet belongs to a location, given that tweet.
```
P(L = l | w1, w2, w3, .... wn) = P(w1, w2, w3, .... wn | L=l) * P(L = l) / P(w1, w2, w3, .... wn)
```
##### Based on below inference of Bayes Law of conditional independency (Naive) :
- probability of occurance of each word does not affect probability of occurance of other words.
```
    P(wi | wj) = P(wi)
```
- since we are going to compare the probabilities, we can ignore the denominator.

So, the problem then boils down to below computation :
```
    P(L = l | w1, w2, ... wn) ∝ P(L = l) ∏ P(wi | L = l) 
    
```


#### Data Structure used to store information given by training data :
```
{
    all_location_count: Integer,
    details: {
        location: {
            location_1: {
                words: [String],
                words_count: Integer,
                tweet_count: Integer,
                location_probability: Float
            },
            .
            .
        }
    }
    all_location: [String],
    all_word_count: Integer
}
```
>This data structure calculates probability P(L=l), tweet count, word count and word array  while walking through the file
>in a single loop, hence avoiding the overhead of calculating them all over again, while doing computations.
>We can access the location probabilities and other values in o(1) time.


#### What we did :
- cleaned the training and test data and removed special characters.
- learnt information from training data, and stored in the earlier explained data structure.
- calculated P(L = l) ∏ P(wi | L = l) of test data, using above information from training data structure.
- compared actual location labels and calculated location labels from test data to find accuracy.
- Wrote the information in a file : classified_data.txt

#### Printing top 5 words for each city :
so, for each location, it boils down to finding :
```
select max 5 of [ P(L = l | wi) ]
```
or, using bayesian formula,
```
select max 5 of [P(wi | L=l)*P(L = l) / P(wi)]
```
> To achieve this task, we created another data structure called bag of words. It is basically a dictionary that contains all the words in the training data as key and their frequency as value. The dictionary helps us to calculate the probability of a occurance of a word in O(1) time.
All other probabilities were earlier implemented when we were doing classification, so, it was just re-using the same functions.
The top 5 words of every location prints on the console.

#### Assumptions:
- The probability of a word not occuring in a location is taken as 10^-9 instead of zero, because training samples are less.
- A tweet that contains a loation as a word, has a high probability of belonging to that location, so the classifier algorithm assigns a high probability to that tweet given that location.

### Running the code:

##### Prerequisites
- Two file in the same location as the code, 'tweets.test1.clean.txt' as test data and 'tweets.train.clean.txt' as train data.
- python 2.7
##### commands:

```
./TweetClassification.py
```
To change appropriate permissions :
```
chomd 755 TweetClassification.py
```

If you get file format errors, then run : 
```
dos2unix TweetClassification.py
```

### Authors
- Shashank Shekhar (shashekh@iu.edu)
- Bivas Maiti (bmaiti@iu.edu)
- Ishneet Singh Arora (iarora@iu.edu)





