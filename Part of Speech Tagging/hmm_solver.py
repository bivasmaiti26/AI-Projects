import math
class Viterbi:
    def __init__(self,sentence):
        self.sentence=sentence
        self.v=[]
        self.tag_tracker=[]
        self.min_probability = pow(10, -50)

    def get_most_probable_tags(self,training_data):
        tags = ['ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']
        for word in self.sentence:
            word=word.lower()
            v_states = {}
            tag_tracker_word={}
            for tag in tags:
                MIN_NUMBER = self.min_probability
                if word not in training_data["emission"]:
                    emission=-9999999 if tag != 'NOUN' else 0
                elif training_data["emission"][word][tag]==0:
                    emission=-9999999
                else:
                    emission = math.log(float(training_data["emission"][word][tag]) / float(training_data['transition'][tag]['_count']+training_data['transition'][tag]['_end']))
                max_last_prob_transition = float('-inf')
                max_tag=''
                if len(self.v) > 0:
                    for prev_tag in tags:
                        if(training_data['transition'][prev_tag][tag]>0):
                            last_state_transition = self.v[len(self.v) - 1][prev_tag] + math.log(float(training_data['transition'][prev_tag][tag])/float(training_data['transition'][prev_tag]['_count']))
                        else:
                            last_state_transition = self.v[len(self.v) - 1][prev_tag]+math.log(MIN_NUMBER / float(training_data['transition'][prev_tag]['_count']))
                        if (max_last_prob_transition < last_state_transition):
                            max_last_prob_transition = last_state_transition
                            max_tag=prev_tag
                    v_states[tag] = emission + max_last_prob_transition
                    tag_tracker_word[tag]=(max_tag)
                else:
                    if(training_data['transition'][tag]['_start']==0):
                        initial_prob=math.log(MIN_NUMBER)
                    else:
                        initial_prob = math.log(float(training_data['transition'][tag]['_start']) /float( training_data['sentences_count']))
                    v_states[tag]=initial_prob+emission
            self.v.append(v_states)
            self.tag_tracker.append(tag_tracker_word)
        return (self.get_sequence(),max(self.v[len(self.sentence)-1].items(), key=lambda item: item[1])[1])

    def get_sequence(self):
        sequence=[]
        max_prob_tag = max(self.v[len(self.sentence)-1].items(), key=lambda item: item[1])[0]
        sequence.append(max_prob_tag)
        for i in range(len(self.sentence)-1,0,-1):
            max_prob_tag=self.tag_tracker[i][max_prob_tag]
            sequence.append(max_prob_tag)
        for i in range(len(self.sentence)):
            sequence[i]=sequence[i].lower() if sequence[i] !='X' else sequence[i]
        return sequence[::-1]




    def calc_posterior(self,training_data,label):
        MIN_NUMBER = self.min_probability
        label_upper=[]
        for pos in label:
            label_upper.append(pos.upper())

        prev_tag = label_upper[0]
        if (training_data['transition'][label_upper[0]]['_start'] == 0):
            posterior = MIN_NUMBER
        else:
            posterior = math.log(float(training_data['transition'][label_upper[0]]['_start']) /float( training_data['sentences_count']))
        if self.sentence[0] not in training_data["emission"]:
            prob_word_given_pos = MIN_NUMBER
        else:
            if training_data['emission'][self.sentence[0]][prev_tag]==0:
                prob_word_given_pos = MIN_NUMBER
            else:
                prob_word_given_pos=float(training_data['emission'][self.sentence[0]][prev_tag])/float(training_data['transition'][prev_tag]['_count'] +training_data['transition'][prev_tag]['_end'])
        posterior+=math.log(prob_word_given_pos)
        for index in range(1,len(label_upper)):
            tag=label_upper[index]
            word=self.sentence[index]
            if training_data['transition'][prev_tag][tag]>0:
                prob_pos_given_prev =float(training_data['transition'][prev_tag][tag])/float(training_data['transition'][prev_tag]['_count'] +training_data['transition'][prev_tag]['_end'])
            else:
                prob_pos_given_prev=MIN_NUMBER
            posterior+=math.log(prob_pos_given_prev)
            if word not in training_data["emission"]:
                prob_word_given_pos=MIN_NUMBER
            else:
                if training_data['emission'][word][tag]>0:
                    prob_word_given_pos=float(training_data['emission'][word][tag])/float(training_data['transition'][tag]['_count'] +training_data['transition'][tag]['_end'])
                else:
                    prob_word_given_pos = MIN_NUMBER
            posterior+=math.log(prob_word_given_pos)
            prev_tag=label_upper[index]
        return posterior

