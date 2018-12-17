import numpy as np
import math
import pickle

class AdaBoost:
    
    def __init__(self, train):
        self.train = train
        self.good_classifier_threshold = 0.69
        self.attempts_to_try = 2000
        self.orientations = (0, 90, 180, 270)
        self.classifiers = {'n_stumps':50,
                            'classifiers': {str(o) : [] for o in self.orientations},
                            'orientation_thresholds': {'0': 15, '90': 15, '180': 15, '270': 15}}
        if self.train is not None:
            self.normalize_weights()
            self.subset_size = int(0.67 * len(self.train))

    def pick_random_dimensions(self):
        return tuple(np.random.randint(0, high = 192, size = 2))

    def pick_random_subset(self):
        return np.random.choice(self.train, size = self.subset_size, p=map(lambda item: item['weight'], self.train))

    def is_classifier_good(self, classifier, orientation):
        error = 0
        train_set = self.pick_random_subset()
        for train_point in train_set:
            h_x_i = self.classify(classifier, train_point, orientation)
            y_i = 1 if train_point['gt_orient'] == orientation else -1
            if h_x_i != y_i:
                error += 1
        if (self.subset_size - error) * 1.0 / self.subset_size > self.good_classifier_threshold:
            return True
        return False

    def normalize_weights(self):
        sum_weights = float(sum([data_point['weight'] for data_point in self.train]))
        for data_point in self.train:
            data_point['weight'] /= sum_weights
    
    def train_classifier(self, classifier, orientation):
        error = 0
        for train_point in self.train:
            y_i = 1 if train_point['gt_orient'] == orientation else -1
            h_x_i = self.classify(classifier, train_point, orientation)
            if h_x_i != y_i:
                error += train_point['weight']
        classifier['alpha'] = self.calculate_alpha(error)
        self.update_weight(error)

    def update_weight(self, error):
        for train_point in self.train:
            train_point['weight'] *= (error / (1 - error))
        self.normalize_weights()

    def classify(self, classifier, data_point, orientation):
        return 1 if data_point['pixels'][classifier['dim'][0]] - data_point['pixels'][classifier['dim'][1]] > self.classifiers['orientation_thresholds'][str(orientation)] else -1

    def calculate_alpha(self, error):
        return 0.5 * math.log((1.0 - error) / error)
                
    def create_and_train_classifiers(self):
        for o in self.orientations:
            for classifier in self.create_classifiers(o):
                self.train_classifier(classifier, o)
                self.classifiers['classifiers'][str(o)].append(classifier)

    def create_classifiers(self, orientation):
        n_classifiers = 0
        classifiers = []
        junk_classifiers = []
        attempt = 0
        while n_classifiers <= self.classifiers['n_stumps']:
            classifier = {'alpha': 0, 'dim': self.pick_random_dimensions()}
            junk_classifiers.append(classifier)
            attempt += 1
            if self.is_classifier_good(classifier, orientation):
                n_classifiers += 1
                classifiers.append(classifier)
            if attempt >= self.attempts_to_try:
                self.classifiers['n_stumps'] = 200
                return junk_classifiers[0 : self.classifiers['n_stumps']]
        return classifiers

    def load_model(self, file):
        with open(file, 'r') as file:
            self.classifiers = pickle.load(file)

    def classifiy_test_point(self, classifier, test_data_point, orientation ):
        return 1 if test_data_point[classifier['dim'][0]] - test_data_point[classifier['dim'][1]] > self.classifiers['orientation_thresholds'][str(orientation)] else -1

    def test(self, test_data_points, output_file):
        output = {'accuracy': None, 'contents':[]}
        correct = 0
        incorrect = 0
        for test_data_point in test_data_points:
            classified = []
            for orientation in self.orientations:
                weighted_sum = 0
                for classifier in self.classifiers['classifiers'][str(orientation)]:
                    weighted_sum += (classifier['alpha'] * self.classifiy_test_point(classifier, test_data_point['pixels'], orientation))
                classified.append({'orientation':orientation, 'val':weighted_sum})
            estimated_label = max(classified, key = lambda item: item['val'])['orientation']
            if estimated_label== test_data_point['gt_orient']:
                correct += 1
                label = test_data_point['gt_orient']
            else:
                incorrect +=1
                label = estimated_label
            output['contents'].append((test_data_point['id'],label))
        output['accuracy'] =  correct*1.0/(correct+incorrect) * 100
        self.write_output_to_file(output_file, output)
        return output

    def write_output_to_file(self, output_file, output):
        with open(output_file, 'w') as file:
            file.write("\n".join(map(lambda item: " ".join(map(lambda it:str(it), item)),output['contents'])))

    def write_model(self, file):
        with open(file, 'w') as file:
            pickle.dump(self.classifiers, file)
