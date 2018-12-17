import json

class DataLoader:
    
    def __init__(self, file_name, flag):
        self.train = []
        self.flag = flag
        self.test = []
        self.file = file_name

    def read(self):
        with open(self.file, 'r') as file:
            for i, line in enumerate(file.readlines()):
                split_line = line.split()
                if self.flag == 0:
                    self.train.append({'id': split_line[0], 'gt_orient': int(split_line[1]), 'weight' : 1,'pixels' : map(lambda item: int(item), split_line[2:len(split_line)])})
                else:
                    self.test.append({'id': split_line[0], 'gt_orient': int(split_line[1]), 'weight': 1, 'pixels': map(lambda item: int(item), split_line[2:len(split_line)])})

    def get_train(self):
        return list(self.train)

    def get_test(self):
        return list(self.test)
    
    def dump(self):
        with open('data_loader_dump.dmp', 'w') as file:
            file.write(json.dumps(self.train if self.flag == 0 else self.test, indent=2))