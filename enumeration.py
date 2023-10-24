import config
import itertools
from genetic_algorithm import total_time

class Enumeration:
    def __init__(self, item_location_mapping):
        self.item_location_mapping = item_location_mapping

    def run(self, order):
        result = []
        permu = []
        for individual in itertools.permutations(order):
            pool = []
            for item in individual:
                pool.append(self.item_location_mapping[item])
            for i in itertools.product(*pool):
                permu.append({
                    'solution': i,
                    'time': total_time(i)
                })
        solution = min(permu, key=lambda s: s.get('time'))
        solution_time = solution.get('time')
        solution_item = solution.get('solution')
        print(solution_item)
        return solution_time

# enum = Enumeration(config.item_location_mapping)
# travel_time_with_enum = enum.run(config.ORDER)
