from __future__ import division

from math import *
from collections import namedtuple, defaultdict
from random import shuffle
from timeit import default_timer
from pprint import pprint

from heapdict import heapdict


WIDTH = 4
HEIGHT = 3

goal = tuple(range(1, WIDTH*HEIGHT)+[0])


def print_state(state):
    for i in range(HEIGHT):
        for j in range(WIDTH):
            print '{:2}'.format(state[i*WIDTH+j]),
        print


def parity(state):
    empty = state.index(0)
    cnt = empty%WIDTH + empty//WIDTH
    for i, x in enumerate(state):
        for y in state[:i]:
            if y > x:
                cnt += 1
    return cnt%2


def adjanced(state):
    empty = state.index(0)

    def move_to(new_empty):
        c = list(state)
        c[empty] = c[new_empty]
        c[new_empty] = 0
        return tuple(c)

    if empty >= WIDTH:
        yield 'u', move_to(empty-WIDTH)
    if empty < WIDTH*(HEIGHT-1):
        yield 'd', move_to(empty+WIDTH)
    if empty % WIDTH > 0:
        yield 'l', move_to(empty-1)
    if empty % WIDTH < WIDTH-1:
        yield 'r', move_to(empty+1)


def num_misplaced(state):
    n = 0
    for x, y in zip(state, goal):
        if x != y and x != 0:
            n += 1
    return n


def manhattan_dist(state):
    dist = 0
    for pos, k in enumerate(state):
        if k == 0:
            continue
        dist += abs(pos % WIDTH - (k-1) % WIDTH)
        dist += abs(pos // WIDTH - (k-1) // HEIGHT)
    return dist


def random_position():
    c = range(WIDTH*HEIGHT)
    shuffle(c)
    return tuple(c)


def a_star(start, heuristic, stats):
    start_time = default_timer()

    closed = set()
    g_cost = {start: 0}
    f_cost = heapdict()
    f_cost[start] = g_cost[start] + heuristic(start)
    prev = {} # dict {state: (prev_state, move)}

    while f_cost:
        stats['states'] += 1
        s, f = f_cost.popitem()
        if s == goal:
            moves = []
            while s != start:
                s, move = prev[s]
                moves.append(move)
            stats['moves'] += len(moves)
            stats['time'] += default_timer()-start_time
            return moves[::-1]

        closed.add(s)
        g2 = g_cost[s] + 1
        for move, s2 in adjanced(s):
            if s2 in closed:
                continue
            if s2 not in f_cost or g_cost[s2] > g2:
                g_cost[s2] = g2
                f_cost[s2] = g2+heuristic(s2)
                prev[s2] = s, move

    stats['time'] += default_timer()-start_time
    return None


def show_path(start, moves):
    s = start
    print_state(s)
    for move in moves:
        print move
        s = dict(adjanced(s))[move]
        print_state(s)


def compare_heuristics():
    heuristics = [
        num_misplaced,
        manhattan_dist,
        row_db_heuristic,
        col_db_heuristic,
    ]

    stats = defaultdict(int)
    for _ in range(100000):
        while True:
            s = random_position()
            if parity(s) == parity(goal):
                break
        m = max(heuristics, key=lambda h: h(s))
        stats[m] += 1
    print dict(stats)


def mask(subset, state):
    return tuple(x if x == 0 or x in subset else -1 for x in state)


def build_template_shard(subset):
    #print mask(subset, goal)

    g = mask(subset, goal)
    queue = [g]
    next_queue = []
    costs = {}

    dist = 0

    while True:
        while queue:
            s = queue.pop()
            if s in costs:
                continue
            costs[s] = dist
            empty = s.index(0)
            for _, s2 in adjanced(s):
                if s2 in costs:
                    continue
                if s2[empty] == -1:
                    queue.append(s2)
                else:
                    next_queue.append(s2)

        if len(next_queue) == 0:
            break
        queue = next_queue
        next_queue = []
        dist += 1

    return costs


def build_template_db(partition):
    assert set.union(*map(set, partition)) == set(range(1, WIDTH*HEIGHT))
    assert sum(map(len, partition)) == WIDTH*HEIGHT-1
    print 'buiding db...',
    result = {}
    for subset in partition:
        result[tuple(subset)] = build_template_shard(subset)
    print 'done,', sum(map(len, result.values())), 'templates'
    return result


def create_db_heuristic(partition):
    # dict {subset: dict {masked state: cost}}
    template_db = build_template_db(partition)

    def db_cost(state):
        result = 0
        for k, v in template_db.items():
            result += v[mask(k, state)]
        return result

    return db_cost


k = 4
xs = range(1, WIDTH*HEIGHT)

partition = [xs[i:i+k] for i in range(0, len(xs), k)]
row_db_heuristic = create_db_heuristic(partition)

m = (len(xs)+k-1)//k
partition = [xs[i::m] for i in range(m)]
col_db_heuristic = create_db_heuristic(partition)


def main():
    compare_heuristics()
    print
    #return

    def zero(s):
        return 0

    def all_db(s):
        return max(row_db_heuristic(s), col_db_heuristic(s))

    heuristics = [
        all_db,
        col_db_heuristic,
        row_db_heuristic,
        #manhattan_dist,
        #num_misplaced
    ]

    for heuristic in heuristics:
        stats = defaultdict(int)
        print heuristic

        n = 0
        time_limit = default_timer()+300
        while default_timer() < time_limit:
            while True:
                s = random_position()
                if parity(s) == parity(goal):
                    break
            moves = a_star(s, heuristic, stats)
            assert moves is not None
            n += 1
        print 'n', n
        #pprint(dict(stats))
        for key in sorted(stats):
            print key, stats[key]/n
        print 'states/s', stats['states']/(stats['time']+1e-6)
        print


if __name__ == '__main__':
    main()