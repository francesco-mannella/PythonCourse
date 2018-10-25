import string
import sys

filename = sys.argv[1]

words = {}

with open(filename, "r") as ml:
    lines = ml.readlines()
    for line in lines:
        line.translate(None, string.punctuation)
        line.translate(None, "\n")
        currWords = line.split()
        for w in currWords:
            if w in words.keys():
                words[w] += 1
            else:
                words[w] = 1

values = sorted(words.values(), reverse=True)
keys = words.keys()

for value in values:
    for key in keys:
        if words[key] == value:
            print("%s: %d" %(key , value))
            keys.remove(key)

