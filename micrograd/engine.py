class Value:

    def __init__(self, data, children=(), _op=''):
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Value(data={self.data})"

a = Value(3.0)
b = Value(-4.0)
c = Value(0.5)
e = a + b
d = e * c
f = Value(-3)
L = d * f

print(f"{L.data}")

d.grad = -3
f.grad = (3 + -4) * 0.5
e.grad = -3 * 0.5
c.grad = -3 * -1
a.grad = -3 * 0.5
b.grad = -3 * 0.5

a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

e = a + b
d = e * c
L = d * f

print(f"{L.data}")