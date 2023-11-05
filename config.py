import sys
import time
import numpy as np

# Genetic Algorithm Parameters
# n
NUMBER_OF_ROWS = 39
# m
NUMBER_OF_COLUMN = 20
# number of aisle
NUMBER_OF_AISLE = 4
# l
LENGTH_OF_STORAGE_BIN = 13.82
# h
HEIGHT_OF_STORAGE_BIN = 7.1
# vy
VERTICAL_VELOCITY = 3
# vx
HORIZONTAL_VELOCITY = 3
# vw
CROSS_VELOCITY = 2
# r
CURVE_LENGTH = 3
# w
RACK_DISTANCE = 4
# sc
INITIAL_SR_AISLE = 0

DENSITY = 0.9

T_H = LENGTH_OF_STORAGE_BIN*NUMBER_OF_COLUMN/HORIZONTAL_VELOCITY

T_V = HEIGHT_OF_STORAGE_BIN*NUMBER_OF_ROWS/VERTICAL_VELOCITY

BIG_T = max([ T_H, T_V ])

SHAPE_FACTOR = min([T_H / BIG_T, T_V / BIG_T])

TIME_RATIO = 1 + (np.power(SHAPE_FACTOR, 2))*1/3

ITEM_NUMERATION = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'apple', 'banana', 'chocolate', 'diamond', 'elephant', 'firefly', 'giraffe', 'happiness', 'igloo', 'jazz', 'koala', 'lemon', 'mountain', 'nautical', 'ocean', 'penguin', 'quasar', 'rainbow', 'sunshine', 'tiger', 'umbrella', 'volcano', 'whisper', 'xylophone', 'zebra', 'astronomy', 'butterfly', 'cinnamon', 'dolphin', 'elegant', 'flamingo', 'guitar', 'harmony', 'island', 'jubilee', 'kangaroo', 'lighthouse', 'marvelous', 'nirvana', 'orchid', 'paradise', 'quokka', 'radiant', 'serendipity', 'tornado', 'unicorn', 'vivid', 'waterfall', 'xenophobe', 'yesterday', 'zephyr', 'adorable', 'ballet', 'charming', 'dandelion', 'ethereal', 'felicity', 'gratitude', 'huckleberry', 'innocence', 'joyful', 'kindness', 'lullaby', 'magnolia', 'nectarine', 'overture', 'precious', 'quaint', 'serenity', 'tranquil', 'uplifting', 'velvet', 'wonderful', 'xenophile', 'yearning', 'zestful', 'aquamarine', 'blossom', 'cosmic', 'dazzling', 'effervescent', 'fantasy', 'glorious', 'heavenly', 'incandescent', 'jubilant', 'kaleidoscope', 'luminous', 'mesmerize', 'nectar', 'opulent', 'paragon', 'quixotic', 'radiance', 'serendipitous', 'tantalizing', 'ubiquitous',
                   'vibrant', 'whimsical', 'xenodochial', 'yield', 'zirconium', 'abracadabra', 'bamboo', 'cascade', 'daffodil', 'eclipse', 'flourish', 'glisten', 'hurricane', 'illusion', 'jasmine', 'kaleidoscope', 'labyrinth', 'mysterious', 'nostalgia', 'opal', 'paradox', 'quicksilver', 'radiate', 'serene', 'twilight', 'unison', 'vortex', 'whisper', 'xylophone', 'yacht', 'zeppelin', 'antelope', 'butterfly', 'chimera', 'dandelion', 'ephemeral', 'fascinate', 'gargantuan', 'hologram', 'inspire', 'jubilee', 'kangaroo', 'labyrinth', 'mystical', 'nebula', 'oasis', 'paragon', 'quintessence', 'rhapsody', 'serendipity', 'talisman', 'ubiquity', 'verdant', 'whimsy', 'xenophile', 'yearn', 'zealot', 'allure', 'bliss', 'cascade', 'dainty', 'elusive', 'felicity', 'gossamer', 'hallowed', 'imagine', 'juxtapose', 'kismet', 'lagoon', 'mythical', 'nirvana', 'oceanic', 'paradise', 'quaint', 'resonate', 'serenity', 'tantalize', 'ubiquitous', 'vivid', 'wonder', 'xanadu', 'yearning', 'zephyr', 'ambrosia', 'breathtaking', 'captivate', 'dreamscape', 'effulgent', 'fantasia', 'grace', 'halcyon', 'inspiration', 'jovial', 'kaleidoscope', 'luminous', 'mellifluous', 'nostalgia', 'oasis', 'paragon', 'quixotic', 'resplendent', 'serendipitous', 'tranquil', 'uplifting', 'vernal', 'whimsical', 'xenodochial', 'yearn', 'zenith']

ORDER = ['A', 'B', 'C', 'D', 'E']

ORDER_LENGTH = len(ORDER)

TEST = [(1, 23, 4), (1, 1, 3), (1, 4, 11), (0, 16, 8)]


current_aisle_of_sr = INITIAL_SR_AISLE

current_aisle_of_sr_for_aco = INITIAL_SR_AISLE

largest_aisle_to_be_visited = 0
smallest_aisle_to_be_visited = 0

aisle_rack_mapping = [(0, 1), (2, 3), (4, 5), (6, 7)]

item_probability = [DENSITY/len(ITEM_NUMERATION)
                    for _ in range(len(set(ITEM_NUMERATION)))]

empty_probability = 1 - sum(item_probability)

print(sys.argv)
if '--use-external-warehouse' in sys.argv:
    warehouse = None
    racks = []
    with open('warehouse.txt', 'r') as rf:
        content = rf.read()
        racks_text = content.split('\n\n')
        for rack in racks_text:
            rows = []
            rows_text = rack.split('\n')
            for row in rows_text:
                columns = row.split(', ')
                rows.append(columns)
            racks.append(rows)
        warehouse = np.array(racks)

else:
    np.random.seed(int(time.time()))
    warehouse = np.random.choice([*set(ITEM_NUMERATION), 0], size=(NUMBER_OF_AISLE*2,
                                 NUMBER_OF_ROWS, NUMBER_OF_COLUMN), p=[*item_probability, empty_probability])
    with open('warehouse.txt', 'w') as f:
        for idx, rack in enumerate(warehouse):
            for ridx, row in enumerate(rack):
                f.write(np.array2string(row, separator=', ', max_line_width=10000, formatter={
                        'str_kind': lambda x: x})[1:-1])
                if ridx != len(rack) - 1:
                    f.write('\n')
            if idx != len(warehouse) - 1:
                f.write('\n\n')

if '--print-warehouse' in sys.argv:
    for idx, rack in enumerate(warehouse):
        for ridx, row in enumerate(rack):
            print(np.array2string(row, separator=', ', max_line_width=10000,
                  formatter={'str_kind': lambda x: x})[1:-1])
        if idx != len(warehouse) - 1:
            print('\n')

index_of_items = list(zip(*np.where(warehouse != '0')))

item_location_mapping = {}

for item in ITEM_NUMERATION:
    item_locations = list(zip(*np.where(warehouse == item)))
    item_location_mapping[item] = item_locations


def get_order_type(item):
    if item == 0:
        return 0
    return warehouse[item[0]][item[1]][item[2]]


def t1(item):
    global current_aisle_of_sr
    total_time = 0
    if item[0] // 2 == current_aisle_of_sr:
        vertical_moving_time = (item[1] + 1) * \
            HEIGHT_OF_STORAGE_BIN / VERTICAL_VELOCITY
        horizontal_moving_time = (
            item[2] + 1) * LENGTH_OF_STORAGE_BIN / HORIZONTAL_VELOCITY
        total_time += 2 * \
            round(max(vertical_moving_time, horizontal_moving_time), 1)
        total_time *= TIME_RATIO

    return total_time


def t2(item):
    global current_aisle_of_sr
    total_time = 0
    if item[0] // 2 != current_aisle_of_sr:
        aisle_travel_time = (NUMBER_OF_COLUMN *
                             LENGTH_OF_STORAGE_BIN) / HORIZONTAL_VELOCITY
        aisle_travel_time *=  TIME_RATIO
        curve_travel_time = 2 * CURVE_LENGTH / CROSS_VELOCITY
        total_time += aisle_travel_time + curve_travel_time
    return total_time


def t3(item):
    global current_aisle_of_sr
    total_time = 0
    if item[0] // 2 != current_aisle_of_sr:
        partial_horizontal_moving_time = (
            NUMBER_OF_COLUMN - item[2] + 1) * LENGTH_OF_STORAGE_BIN / HORIZONTAL_VELOCITY
        partial_horizontal_moving_time *= TIME_RATIO
        vertical_moving_time = (item[1] + 1) * \
            HEIGHT_OF_STORAGE_BIN / VERTICAL_VELOCITY
        horizontal_moving_time = (
            item[2] + 1) * LENGTH_OF_STORAGE_BIN / HORIZONTAL_VELOCITY
        total_time += partial_horizontal_moving_time + \
            max(vertical_moving_time, horizontal_moving_time) * TIME_RATIO
        current_aisle_of_sr = item[0] // 2
    return total_time


def t4():
    global largest_aisle_to_be_visited
    global smallest_aisle_to_be_visited
    distance = RACK_DISTANCE * (
        largest_aisle_to_be_visited - smallest_aisle_to_be_visited
        + min(abs(current_aisle_of_sr - smallest_aisle_to_be_visited),
              abs(largest_aisle_to_be_visited - current_aisle_of_sr))
    )
    total_time = distance / CROSS_VELOCITY
    return total_time

# Objective Function


def total_t(solution, reset_aisle=True):
    global current_aisle_of_sr
    global largest_aisle_to_be_visited
    global smallest_aisle_to_be_visited

    if len(set(solution)) < len(solution):
        return float('inf')

    if reset_aisle:
        current_aisle_of_sr = INITIAL_SR_AISLE

    try:
        time = 0
        maximum_rack_number = max(solution, key=lambda x: x[0])[0]
        minimum_rack_number = min(solution, key=lambda x: x[0])[0]

        largest_aisle_to_be_visited = maximum_rack_number // 2
        smallest_aisle_to_be_visited = minimum_rack_number // 2
        cross_time = t4()
        for item in solution:
            time += t1(item) + t2(item) + t3(item)
        time += cross_time
        return time
    except Exception as e:
        print(e)
        print(solution)

# For debugging


if '--test' in sys.argv:
    for item in TEST:
        print(warehouse[item[0]][item[1]][item[2]])
    print(total_t(TEST))
    exit()

if '--shape' in sys.argv:
    print(SHAPE_FACTOR)
    exit()

def exit(message=''):
    print(message)
    return sys.exit()
