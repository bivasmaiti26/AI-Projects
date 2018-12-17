import numpy as np
import pickle
import collections

class Nearest:
    def __init__(self, trainer, tester):
        self.model=trainer
        self.tester=tester
        self.k = 71

    def test_knn(self,test,output_file):
        correct = 0
        incorrect  = 0
        output = {'accuracy' : None, 'content':[]}
        for test_point in test:
            eucl_dist = self.get_distance(test_point['pixels'])
            sorted_ecul_distance = sorted(eucl_dist, key = lambda item : item[1])
            k_nearest_neighbours = sorted_ecul_distance[0:self.k]
            votes = [i[0] for i in k_nearest_neighbours]
            vote_result = collections.Counter(votes).most_common(1)[0][0]
            if vote_result == test_point['gt_orient']:
                correct += 1
            else:
                incorrect += 1
            output['content'].append((test_point['id'], vote_result))
        output['accuracy'] = correct * 100.0 / (correct + incorrect)
        self.write_output_to_file(output_file, output)
        return output

    def write_model(self, model_file):
        with open(model_file, 'wb') as file:
            pickle.dump(self.model[0:16000], file)

    def load_model(self, model_file):
        with open(model_file, 'r') as file:
            self.model = pickle.load(file)
        self.model = map(lambda item: {'id': item['id'], 'gt_orient': item['gt_orient'], 'pixels': np.array(item['pixels'])}, self.model)

    def write_output_to_file(self, output_file, output):
        with open(output_file, 'w') as file:
            file.write("\n".join(map(lambda item: " ".join(map(lambda it: str(it), item)), output['content'])))


    def get_distance(self, test_pixel_vector):
        return [ (data_point['gt_orient'],np.linalg.norm(data_point['pixels'] - test_pixel_vector)) for data_point in self.model]


