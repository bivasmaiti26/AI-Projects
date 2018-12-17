#!/bin/python

from data_loader import DataLoader
from adaboost import AdaBoost
from forest import Forest
from nearest import Nearest
import sys
import time

class Orient:
    
    def __init__(self, flag, model_file, model, train_test_file):
        self.flag = flag
        self.model_file = model_file
        self.model = model
        self.train_test_file = train_test_file
        self.output_file = 'adaboost_output.txt' if model == 'adaboost' else 'knn_output.txt' if model == 'nearest' else 'forest_output.txt' if model == 'forest' else 'best_output.txt'

    def load_data(self):
        data_loader_instance = DataLoader(self.train_test_file, self.flag)
        data_loader_instance.read()
        if self.flag == 0:
            self.train = data_loader_instance.get_train()
        else:
            self.test = data_loader_instance.get_test()

    def do_adaboost(self):
        if self.flag == 0:
            start = time.time()
            adaboost_instance = AdaBoost(self.train)
            adaboost_instance.create_and_train_classifiers()
            adaboost_instance.write_model(self.model_file)
            end = time.time()
            print 'Training Time :', (end - start)/60, 'mins'
        else:
            start = time.time()
            adaboost_instance = AdaBoost(None)
            adaboost_instance.load_model(self.model_file)
            test_output = adaboost_instance.test(self.test, self.output_file)
            print test_output['accuracy'], '%'
            end = time.time()
            print 'Testing Time :', (end - start)/60, 'mins'

    def do_forest(self):
        if self.flag == 0:
            start = time.time()
            forest_instance = Forest(self.train, None)
            forest_instance.build_forest()
            forest_instance.write_model(self.model_file)
            end = time.time()
            print 'Training Time :', (end - start)/60, 'mins'
        else:
            start = time.time()
            forest_instance = Forest(None, self.test)
            forest_instance.load_model(self.model_file)
            test_output = forest_instance.test_forest(self.test, self.output_file)
            print test_output['accuracy'], '%'
            end = time.time()
            print 'Testing Time :', (end - start)/60, 'mins'

    def do_knn(self):
        if self.flag == 0:
            start = time.time()
            nearest_instance=Nearest(self.train,None)
            nearest_instance.write_model(self.model_file)
            end = time.time()
            print 'Training Time :', (end - start) / 60, 'mins'
        else:
            start = time.time()
            nearest_instance = Nearest(None, self.test)
            nearest_instance.load_model(self.model_file)
            test_output = nearest_instance.test_knn(self.test, self.output_file)
            print test_output['accuracy'], '%'
            end = time.time()
            print 'Testing Time :', (end - start) / 60, 'mins'

    def orient(self):
        self.load_data()
        if self.model == 'adaboost':
            self.do_adaboost()
        elif self.model == 'nearest':
            self.do_knn()
        elif self.model=='forest':
            self.do_forest()
        elif self.model=='best':
            self.do_knn()


def init():
    flag = 0 if 'train' == sys.argv[1] else 1 if 'test' == sys.argv[1] else -1
    train_test_file = sys.argv[2]
    model_file = sys.argv[3]
    model = sys.argv[4]

    if len(sys.argv) != 5:
        print 'argument length needs to be 4'
        sys.exit(1)

    if flag == -1:
        print 'wrong flag at argument 1'
        sys.exit(1)

    if model not in ['nearest', 'adaboost', 'forest', 'best']:
        print 'invalid model at argument 4'
        sys.exit(1)

    Orient(flag, model_file, model, train_test_file).orient()


init()
