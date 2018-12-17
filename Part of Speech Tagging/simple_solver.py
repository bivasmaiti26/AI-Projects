# !/usr/bin/python
import math
class Simplified:
    def __init__(self, sentence):
        self.sentence = sentence
        self.v = []
        self.tag_tracker = []

    def get_most_probable_tags(self, training_data):
        sequence = []
        # Write code to change value of sequence here!!
        for word in self.sentence:
            maxvar = -1
            maxtag = ''
            if word in training_data["emission"]:
                for each in training_data["emission"][word]:
                    if each != "_count":
                        a = float((training_data["emission"][word][each])) / training_data["emission"][word]["_count"]
                        if maxvar < a:
                            maxvar = a
                            maxtag = each
            else:
                maxtag = 'NOUN'
            sequence.append(maxtag.upper())
        # use below snippet to convert to lowercase then return
        for i in range(len(self.sentence)):
            sequence[i] = sequence[i].lower() if sequence[i] != 'X' else sequence[i]
        return sequence

    def calc_posterior(self, training_data, label):
        tags = ['ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']
        label_upper = []
        posterior = 1
        i = 0
        count = 0;
        for i in range(len(label)):
            label_upper.append(label[i].upper())
            word = self.sentence[i]
            b = label[i].upper()
            if word in training_data["emission"]:
                
                # qwer=math.log(2)
                z = float((training_data["emission"][word][b]))
                zx = float(training_data["transition"][b]["_count"] + training_data["transition"][b]["_end"])
                if z and zx:
                    posterior = posterior + math.log(z) - math.log(zx)
                    count = count + 1;
            else:
                prob_pos_given_prev = .000000001
                posterior += math.log(prob_pos_given_prev)
        return posterior