import config
import itertools
from genetic_algorithm import total_time

class Enumeration:
    def __init__(self, item_location_mapping):
        self.item_location_mapping = item_location_mapping

    def run(self, order):
        result = []
        permu = []
        for item in order:
            result.append(self.item_location_mapping[item])
        for i in itertools.product(*result):
            permu.append(total_time(i))
        time = min(permu)
        return time

# enum = Enumeration(config.item_location_mapping)
# travel_time_with_enum = enum.run(config.ORDER)
