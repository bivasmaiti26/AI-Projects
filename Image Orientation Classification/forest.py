import collections
import math
import numpy as np
import pickle

class Forest:
    def __init__(self,trainer,tester):
        self.trainer=trainer
        self.tester=tester
        self.model=[]
        self.subsets={}
        self.num_trees=100
        self.num_predicates=192
        self.orient_labels=[0,90,180,270]
        self.threshold=128
        self.purity=0.9
        self.dimensions=14

    def build_tree(self,subset,subset_attributes):
        count_0=0
        count_90=0
        count_180=0
        count_270=0
        len_subset=len(subset)
        for datapoint in subset:
            count_0 += 1 if datapoint['gt_orient'] == 0 else 0
            count_90 += 1 if datapoint['gt_orient'] == 90 else 0
            count_180 += 1 if datapoint['gt_orient'] == 180 else 0
            count_270 += 1 if datapoint['gt_orient'] == 270 else 0
        if count_0>len_subset* self.purity:
            return {0:[None,None]}
        elif count_90>len_subset* self.purity:
            return {90:[None,None]}
        elif count_180>len_subset* self.purity:
            return {180:[None,None]}
        elif count_270>len_subset* self.purity:
            return {270:[None,None]}
        elif len(subset_attributes)==1:
            #split data on only predicate
            chosen_pred=subset_attributes[0]
            left_subset = [datapoint for datapoint in subset if datapoint['pixels'][chosen_pred] < self.threshold]
            right_subset = [datapoint for datapoint in subset if datapoint['pixels'][chosen_pred] >= self.threshold]
            return {chosen_pred:[self.build_tree(left_subset,[]),self.build_tree(right_subset,[])]}
        elif len(subset_attributes)==0 or len(subset)==0:
            if max(count_0,count_90,count_180,count_270)==count_0:
                return {0: [None,None]}
            elif max(count_0,count_90,count_180,count_270)==count_90:
                return {90: [None,None]}
            elif max(count_0,count_90,count_180,count_270)==count_180:
                return {180: [None,None]}
            elif max(count_0,count_90,count_180,count_270)==count_270:
                return {270: [None,None]}
        else:
            chosen_pred=self.find_least_entropy_predicate(subset,subset_attributes)
        remaining_predicates = [pred for pred in subset_attributes if pred != chosen_pred]
        left_subset=[datapoint for datapoint in subset if datapoint['pixels'][chosen_pred]<self.threshold ]
        right_subset = [datapoint for datapoint in subset if datapoint['pixels'][chosen_pred] >= self.threshold]
        return {chosen_pred: [self.build_tree(left_subset,remaining_predicates),self.build_tree(right_subset, remaining_predicates)]}

    def find_least_entropy_predicate(self,data_points,predicates):
        min_disorder=float('inf')
        min_disorder_pred=None
        data_points_num=len(data_points)
        sum_pred = 0.0
        for pred in predicates:
            # mean is tao or threshold
            pred_branches = [True,False]
            #loop for future. If time permits add more branches
            for branch in pred_branches:
                num_branch = 0.0
                for i in range(data_points_num):
                    #if less than mean, branch-true, else false
                    num_branch += 1 if (data_points[i]['pixels'][pred] < self.threshold )== branch else 0
                #print num_branch,data_points_num

                sum_branch=0.0
                for orient in self.orient_labels:
                    num_branch_orient=0.0
                    #calculate n[i,c] for each branch and class
                    for i in range(data_points_num):
                        num_branch_orient+=1 if ((data_points[i]['pixels'][pred]<self.threshold)==branch) and data_points[i]['gt_orient']==orient else 0
                    if num_branch_orient==0 or num_branch==0:
                        sum_branch+=0
                    else:
                        sum_branch+=(float(-num_branch_orient)/float(num_branch))*(math.log(float(num_branch_orient)/float(num_branch),2))
                       # print (float(-num_branch_orient)/float(num_branch)),math.log(float(num_branch_orient)/float(num_branch),2)
                if not (sum_branch==0 or data_points_num==0):
                    sum_pred+=(float(num_branch)/float(data_points_num))*sum_branch
                else:
                    sum_pred+=0
            if min_disorder>sum_pred:
                min_disorder=sum_pred
                min_disorder_pred=pred
        if min_disorder<0:
            print min_disorder
        return min_disorder_pred

    def test_forest(self,  test_data, output_file):
        correct=0
        images=0
        output = {'accuracy': None, 'content': []}
        for image in test_data:
            labels=[]
            images+=1
            for tree in self.model:
                labels.append(self.predict_orient(image['pixels'],tree))
            label=max(collections.Counter(labels),key=collections.Counter(labels).get)
            if label==image['gt_orient']:
                correct+=1
            output['content'].append((image['id'], label))
        output['accuracy'] =  float(correct)/float(images) * 100
        self.write_output_to_file(output_file, output)
        return output

    def predict_orient(self,image,tree):
        if tree ==None:
            print "random choice"
            return np.random.choice([0,90,180,270])
        predicate=[item for item in tree.keys()][0]
        if [None,None] in tree.values():
            return predicate
        if image[predicate]<self.threshold:
            return self.predict_orient(image,tree[predicate][0])
        else:
            return self.predict_orient(image,tree[predicate][1] )
    def get_two_thirds_data(self,data):
        len_data=len(data)
        random_indices=list(np.random.random_integers(0, len_data - 1, (2*len_data)/3))
        return [data[i] for i in random_indices]

    def build_forest(self):
        for i in range(self.num_trees):
            subset_0= [datapoint for datapoint in self.trainer if datapoint['gt_orient']==0]
            subset_90 = [datapoint for datapoint in self.trainer if datapoint['gt_orient'] == 90]
            subset_180 = [datapoint for datapoint in self.trainer if datapoint['gt_orient'] == 180]
            subset_270 = [datapoint for datapoint in self.trainer if datapoint['gt_orient'] == 270]
            subset=self.get_two_thirds_data(subset_0)+self.get_two_thirds_data(subset_90)+self.get_two_thirds_data(subset_180)+self.get_two_thirds_data(subset_270)
            subset_attributes = np.random.random_integers(0, self.num_predicates - 1, self.dimensions)
            first_pred=self.find_least_entropy_predicate(subset, subset_attributes)
            remaining_predicates = [pred for pred in subset_attributes if pred != first_pred]
            root={first_pred:{ 'left':None,'right':None,'orient':None}}
            left_subset = [datapoint for datapoint in subset if datapoint['pixels'][first_pred] < self.threshold]
            right_subset = [datapoint for datapoint in subset if datapoint['pixels'][first_pred] >= self.threshold]
            tree={first_pred:[self.build_tree(left_subset,remaining_predicates),self.build_tree(right_subset, remaining_predicates)]}
            self.model.append(tree)
        return self.model

    def write_model(self, model_file):
        with open(model_file, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, model_file):
        with open(model_file, 'r') as file:
            self.model = pickle.load(file)

    def write_output_to_file(self, output_file, output):
        with open(output_file, 'w') as file:
            file.write("\n".join(map(lambda item: " ".join(map(lambda it:str(it), item)),output['content'])))