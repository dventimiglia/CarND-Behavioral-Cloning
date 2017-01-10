from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def file_generator(f, func=lambda x:x):
    while True:
        line = f.readline()
        if line=='':
            f.seek(0)
            line = f.readline()
        line = line.rstrip("\r\n")
        yield func(line)
with open("test.txt") as f:
    filegen = file_generator(f, func=lambda x: len(x))
    batchgen = grouper(filegen, 5)
    for l in batchgen:
        print(l)

