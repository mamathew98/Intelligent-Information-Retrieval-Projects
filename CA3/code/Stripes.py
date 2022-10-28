import collections
import functools
import itertools
import operator
import re
from collections import Counter
from collections import defaultdict

from mrjob.job import MRJob
from mrjob.step import MRStep

# Importing the library
import psutil
import time

WORD_RE = re.compile(r"[\w']+")


class MRStripes(MRJob):

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper1,
                # combiner=self.combiner1,
                reducer=self.reducer1),
            MRStep(
                reducer=self.reducer2),
            MRStep(
                reducer=self.reducer_sort),
        ]

    totals = Counter()

    def mapper1(self, _, line):
        words = WORD_RE.findall(line)
        for a in words:
            stripe = Counter()
            for b in words:
                if a != b:
                    stripe[b.lower()] += 1
            yield a.lower(), stripe

    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    def reducer1(self, key, values):
        stripe = Counter()

        for value in values:
            for token, count in value.items():
                stripe[token] += count
                self.totals[token] += count

        yield key, stripe

    def reducer2(self, key, values):
        for value in values:
            for token, count in value.items():
                freq = round(count / self.totals[key], 3)
                yield None, (freq, (key, token))

    def reducer_sort(self, _, relative_freq):
        top100 = itertools.islice(sorted(relative_freq, reverse=True), 100)
        for frequency, pair in top100:
            yield pair, frequency

if __name__ == '__main__':
    start_time = time.time()

    MRStripes.run()

    print("--- %s seconds ---" % (time.time() - start_time))