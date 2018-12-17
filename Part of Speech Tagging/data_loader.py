import re
import json


class Trainer_builder:
    def __init__(self, train_file_name):
        self.tags = ['ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']
        self.train_file_name = train_file_name
        self.trainer = {'sentences_count': 0, 'transition': {},'word_count':0, 'emission': {}, 'double_transition' :{}, 'reverse_emission': {}, 'initial_probability': {t: 0 for t in self.tags+['_count']} }
        self.read()
        #self.dump(self.trainer)

    def read(self):
        with open(self.train_file_name, 'r') as file:
            for line in file.readlines():
                self.trainer['sentences_count']+=1
                self.build_trainer(line)

    def build_trainer(self, line):
        for group_count, (word, tag) in enumerate(re.findall(r'(.+?) (.+?) ', line)):
            self.trainer['word_count']+=1
            word=word.lower()
            if group_count==0 :
                self.trainer['initial_probability'][tag] +=1
                self.trainer['initial_probability']['_count'] += 1

            if self.trainer['emission'].get(word, False) == False:
                self.trainer['emission'][word] = {t: 1 if t == tag else 0 for t in self.tags}
                self.trainer['emission'][word]['_count'] = 1
            else:
                self.trainer['emission'][word][tag] += 1
                self.trainer['emission'][word]['_count'] += 1

            if self.trainer['reverse_emission'].get(tag, False) == False:
                self.trainer['reverse_emission'][tag] = {}
                self.trainer['reverse_emission'][tag]['_count'] = 1
            else :
                self.trainer['reverse_emission'][tag]['_count'] += 1
                if self.trainer['reverse_emission'][tag].get(word, False) == False :
                    self.trainer['reverse_emission'][tag][word] = 1
                else:
                    self.trainer['reverse_emission'][tag][word] += 1

        tags_of_line = self.tags_of_line(line)
        length_tags_of_line = len(tags_of_line)

        for tag_index in range(0, length_tags_of_line):
            if self.trainer['transition'].get(tags_of_line[tag_index], False) == False:
                self.trainer['transition'][tags_of_line[tag_index]] = {t: 0 for t in self.tags}
                self.trainer['transition'][tags_of_line[tag_index]]['_count'] = 1
                self.trainer['transition'][tags_of_line[tag_index]]['_start'] = 1 if tag_index == 0 else 0
                self.trainer['transition'][tags_of_line[tag_index]]['_end'] = 1 if tag_index == length_tags_of_line-1 else 0
                if tag_index + 1 != length_tags_of_line :
                    self.trainer['transition'][tags_of_line[tag_index]][tags_of_line[tag_index + 1]] = 1
            else :
                if tag_index + 1 != length_tags_of_line :
                    self.trainer['transition'][tags_of_line[tag_index]][tags_of_line[tag_index + 1]] += 1
                    self.trainer['transition'][tags_of_line[tag_index]]['_count'] += 1
                    self.trainer['transition'][tags_of_line[tag_index]]['_start'] += 1 if tag_index == 0 else 0
                else:
                    self.trainer['transition'][tags_of_line[tag_index]]['_end'] += 1

        for tag_index in range(0, length_tags_of_line):
            if tag_index + 1 != length_tags_of_line :
                if self.trainer['double_transition'].get(tags_of_line[tag_index]+"_"+tags_of_line[tag_index+1], False) == False:
                    self.trainer['double_transition'][tags_of_line[tag_index]+"_"+tags_of_line[tag_index+1]] = {t: 0 for t in self.tags}
                    self.trainer['double_transition'][tags_of_line[tag_index]+"_"+tags_of_line[tag_index+1]]['_count'] = 1
                    if tag_index + 2 != length_tags_of_line :
                        self.trainer['double_transition'][tags_of_line[tag_index]+"_"+tags_of_line[tag_index+1]][tags_of_line[tag_index + 2]] = 1
                else :
                    if tag_index + 2 != length_tags_of_line :
                        self.trainer['double_transition'][tags_of_line[tag_index]+"_"+tags_of_line[tag_index+1]][tags_of_line[tag_index + 2]] += 1
                        self.trainer['double_transition'][tags_of_line[tag_index]+"_"+tags_of_line[tag_index+1]]['_count'] += 1




    def tags_of_line(self, line):
        tags_from_line = []
        found_dot = False
        for word in line.split():
            if word in self.tags:
                if not found_dot and word == '.':
                    tags_from_line.append(word)
                elif word != '.':
                    tags_from_line.append(word)
                found_dot = word == '.'
        return tags_from_line

    def dump(self, type):
        with open('trainer.dat', 'w') as file:
            file.write(json.dumps(self.trainer, indent=2))

    def get_trainer(self):
        return self.trainer
