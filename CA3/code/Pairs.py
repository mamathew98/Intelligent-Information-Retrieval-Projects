
from mrjob.job import MRJob
from mrjob.step import MRStep

import re
from collections import defaultdict
from itertools import combinations
import itertools

# Importing the library
import psutil
import time



WORD_RE = re.compile(r"[\w']+")


class MRPiars(MRJob):

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_make_tuples,
                combiner=self.combiner_count_sums,
                reducer=self.reducer_count_pairs),
            MRStep(
                reducer=self.reducer_frequency),
            MRStep(
                reducer=self.reducer_sort)
        ]

    totals = defaultdict(int)

    def mapper_make_tuples(self, _, line):
        words = WORD_RE.findall(line)
        for a in words:
            for b in words:
                if a != b:
                    yield (a.lower(), "*"), 1
                    yield (a.lower(), b.lower()), 1

    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    def combiner_count_sums(self, pair, counts):
        yield pair, sum(counts)

    def reducer_count_pairs(self, pair, counts):
        count = sum(counts)
        a, b = pair
        if b == "*":
            self.totals[a] = count
        else:
            yield (a, b), count

    def reducer_frequency(self, pair, countY):
        a, b = pair
        denominator = self.totals[a]

        yield None, (round((sum(countY) / denominator), 3), pair)

    def reducer_sort(self, _, relative_freq):
        top100 = itertools.islice(sorted(relative_freq, reverse=True), 100)
        for frequency, pair in top100:
            yield pair, frequency


if __name__ == '__main__':
    start_time = time.time()

    MRPiars.run()

    print("--- %s seconds ---" % (time.time() - start_time))