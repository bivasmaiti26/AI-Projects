#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2018)
#

from PIL import Image, ImageDraw, ImageFont

import sys
import sys
import math
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print im.size
    print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    #print TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) )
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[2] ])
text=[]
per=[]
def initial_prob(text,frequency):
    count=len(text)
    for i in text:
        if (not i in frequency):  # and (i in symbols):
            frequency[i] = 0
        frequency[i] += float(1)/count
    #frequency=float(frequency)/count
    return (frequency)

#Reading the file and storing the output in text
f = open("test-strings.txt", "r")
for line in f:
    per.append(line)
    text.append(line[0])

frequency={}
#per=[]
a=initial_prob(text,frequency)
#print(a)
#b=initial_prob(text2)
#print(b)


def emission(testcharindex):
    characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    match=0
    didnt=0
    trains = len(train_letters)
    tests = len(test_letters)
    temp={}
    emissions = np.zeros(shape=(trains,tests))
    for letter in characters:
        val = train_letters.get(letter)
        #print val
        obs = test_letters[testcharindex]
        didnt = 0
        match = 1
        for i in range(len(obs)):
            if obs[i] == val[i]:
                #match += 1
                match*=0.8
            else:
  #      	    didnt += 1
		    match*=0.2
        	temp[letter] =float(match)# / (match + didnt)
	return(temp)

#a=emission(0)
#print(a)
emission_final={}
for i in range(0,len(test_letters)):
#	print(i)
	a=emission(i)
#	print(a)
	emission_final[i]=a
#print(emission_final)
#print frequency
transition=[]
p=len(per)
for i in range(p):
    for j in range(len(per[i])-1):
        #print(per[i][j],per[i][j+1])
        a=per[i][j],per[i][j+1]
 #       print(a)
        transition.append(a)

#print(transition)
#frequency.clear()
frequency2={}
b=initial_prob(transition,frequency2)
print(b)

Output_final={}
Output={}
for j in range(len(test_letters)):
    if j==0:
        for i in frequency:
        #print(frequency[i])
	#	print(emission_final[0][i])
			a=(emission_final[0][i])*(frequency[i])
		#print(a)
			Output[i]=a
			Output_final[j]=Output
print(max(Output,key=Output.get))
print(Output_final)
	#if (j==1):
	#	for i in range(len(train_letters)):
		#print("HEY")
	#		print(Output_final[j-1])
	#else:break
#print(Output)
#print(Output_final[0])


trains = len(train_letters)
tests = len(test_letters)
#print(Output)
#for i in range(1,tests):
#	for j in range(trains):
#		for k in range(trains):
			
       #print(per[i][j],per[i][j+1])
