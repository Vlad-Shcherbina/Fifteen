from __future__ import division

from math import *
from collections import namedtuple, defaultdict
from random import shuffle
from timeit import default_timer
from pprint import pprint

from heapdict import heapdict


WIDTH = 4
HEIGHT = 3

State = namedtuple('State', 'cells empty')

goal = State(tuple(range(1, WIDTH*HEIGHT)+[0]), WIDTH*HEIGHT-1)


def print_state(state):
    for i in range(HEIGHT):
        for j in range(WIDTH):
            print '{:2}'.format(state.cells[i*WIDTH+j]),
        print


def parity(state):
    cnt = state.empty%WIDTH + state.empty//WIDTH
    for i, x in enumerate(state.cells):
        for y in state.cells[:i]:
            if y > x:
                cnt += 1
    return cnt%2


def adjanced(state):
    def move_to(new_empty):
        c = list(state.cells)
        c[state.empty] = c[new_empty]
        c[new_empty] = 0
        return State(tuple(c), new_empty)

    if state.empty >= WIDTH:
        yield 'u', move_to(state.empty-WIDTH)
    if state.empty < WIDTH*(HEIGHT-1):
        yield 'd', move_to(state.empty+WIDTH)
    if state.empty % WIDTH > 0:
        yield 'l', move_to(state.empty-1)
    if state.empty % WIDTH < WIDTH-1:
        yield 'r', move_to(state.empty+1)
    

def num_misplaced(state):
    n = 0
    for x, y in zip(state.cells, goal.cells):
        if x != y and x != 0:
            n += 1
    return n


def manhattan_dist(state):
    dist = 0
    for pos, k in enumerate(state.cells):
        if k == 0:
            continue
        dist += abs(pos % WIDTH - (k-1) % WIDTH)
        dist += abs(pos // WIDTH - (k-1) // HEIGHT)
    return dist


def random_position():
    c = range(WIDTH*HEIGHT)
    shuffle(c)
    return State(tuple(c), c.index(0))


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


def main():
    def zero(s):
        return 0

    for heuristic in manhattan_dist, num_misplaced: # , zero:
        stats = defaultdict(int)
        print heuristic

        n = 0
        time_limit = default_timer()+1000
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
        print


if __name__ == '__main__':
    main()