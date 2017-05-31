from itertools import chain

class SmtFormula(object):
    def __init__(self):
        self.lines = []
        self.declarations = []
        self.directives = []

    def __str__(self):
        return "\n".join(chain(map(str, self.declarations),
                               map(str, self.lines),
                               map(str, self.directives)))

    def declare(self, name, type, args="()"):
        self.declarations.append("(declare-fun {!s} {args} {!s})".format(name, type, args=args))

    def assert_(self, val):
        self.lines.append("(assert {!s})".format(val))

    def minimize(self, val):
        self.directives.append("(minimize {!s})".format(val))

    def check_sat(self):
        self.directives.append("(check-sat)")



class Op(object):
    def __init__(self, name, *params):
        self.name = name
        self.params = params

    def __str__(self):
        return "("+ str(self.name) + ' ' + ' '.join(map(str, self.params)) + ')'


    def __eq__(self, other):
        return self._associative("=", other)

    def __ne__(self, other):
        return ~(self == other)


    def __add__(self, other):
        return self._associative("+", other)

    def __sub__(self, other):
        return Op("-", self, other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return self._associative("*", other)

    def __rmul__(self, other):
        return  self._associative("*", other)

    def __and__(self, other):
        return self._associative("and", other)

    def __or__(self, other):
        return self._associative("or", other)

    def __neg__(self):
        return Op("-", self)

    def __invert__(self):
        return Op("not", self)

    def __ge__(self, other):
        return Op(">=", self, other)
    def __le__(self, other):
        return Op("<=", self, other)
    def __gt__(self, other):
        return Op(">", self, other)
    def __lt__(self, other):
        return Op("<", self, other)

    def _associative(self, oper, other):
        if self.name == oper:
            return Op(oper, *(self.params + tuple([other])))
        else:
            return Op(oper, self, other)

class Symbol(Op):
    def __init__(self, name, *paras):
        super(Symbol, self).__init__(name.format(*paras), [])
    def __str__(self):
        return str(self.name)

    def __call__(self, *args):
        return Op(self.name, *args)
