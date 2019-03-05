from re import split


def parse_2in4(f):
    ret = []
    for line in f:
        if not line.startswith("TWOINFOUR"): continue
        res = list(map(int, split(r'[^0-9]+', line)[1:-1]))
        assert len(res) == 4, res
        a, b, c, d = res
        ret.append(res)
    return ret