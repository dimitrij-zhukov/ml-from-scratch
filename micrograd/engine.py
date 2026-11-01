class Value:

    def __init__(self, data, children=(), _op=''):
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.grad = 0.0

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
        
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __repr__(self):
        return f"Value(data={self.data})"

res = (Value(3.0) + Value(-1.5)) * Value(4.0)
print(f'{res.data}')