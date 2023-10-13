import config
import itertools
from itertools import permutations 

class Enumeration:
    def __init__(self, item_location_mapping):
        self.item_location_mapping = item_location_mapping

    def run(self, order):
        result = []
        for item in order:
            result.append(self.item_location_mapping[item])
            print(len(self.item_location_mapping[item]))
        # print(result)
        # permu = list(itertools.product(*result))
        # print(permu)

enum = Enumeration(config.item_location_mapping)
enum.run(config.ORDER)
