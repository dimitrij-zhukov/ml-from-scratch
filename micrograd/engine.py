class Value:

    def __init__(self, data, children=(), _op=''):
        self.data = data
        self._prev = set(children)
        self._op = _op

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
        
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    # def __repr__(self):

print(Value(2.0) * Value(-3.0) + Value(10.0))
res = Value(2.0) * Value(-3.0) + Value(10.0)
print(f'{res.data}')