import os
import json

class InternalDataLoader:

    def __init__(self, config):
        self.config = config

    def load_internal_data(self, load_internal_file_name):

        if not os.path.isfile(load_internal_file_name):
            raise ("[!] Data %s not found" % load_internal_file_name)

        with open(load_internal_file_name, 'r') as file:
            for line in file:
                sentence = json.loads(line)
                print("sentence ", sentence)
                print(sentence['sentence_id'])
