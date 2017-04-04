from itertools import chain

class SmtFormula(object):
    def __init__(self):
        self.lines = []
        self.declarations = []

    def __str__(self):
        return "\n".join(chain(map(str, self.declarations), map(str,self.lines)))

    def declare(self, name, type):
        self.declarations.append("(declare-fun {} () {})".format(name, type))

    def assert_(self, val):
        self.lines.append("(assert {})".format(val))



class Op(object):
    def __init__(self, name, *params):
        self.name = name
        self.params = params

    def __str__(self):
        return "("+ str(self.name) + ' ' + ' '.join(map(str, self.params)) + ')'


    def __eq__(self, other):
        return self._associative("=", other)

    def __ne__(self, other):
        return Op("!=", self, other)

    def __add__(self, other):
        return self._associative("+", other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return self._associative("*", other)

    def __and__(self, other):
        return self._associative("and", other)

    def __or__(self, other):
        return self._associative("or", other)

    def __neg__(self):
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
            return Op(oper, *(self.params + [other]))
        else:
            return Op(oper, self, other)

class Symbol(Op):
    def __init__(self, name):
        super(Symbol, self).__init__(name, [])
    def __str__(self):
        return str(self.name)

