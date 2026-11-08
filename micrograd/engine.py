import math

class Value:

    def __init__(self, data, children=(), _op=''):
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.grad = 0.0

        def _backward():
            return
        self._backward = _backward
        

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')

        def _backward():
            self.grad += out.grad * (other.data * (self.data ** (other.data - 1)))

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __sub__(self, other):
        return self + -(other)

    def __rsub__(self, other):
            return -(self) + other

    def __neg__(self):
        return self * -1

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - out.data**2)

        out._backward = _backward
        return out

    def backward(self):
        ls = []
        visited = set()
        def topo(v):
            if v in visited:
                return
            for el in v._prev:
                topo(el)
            ls.append(v)
            
            visited.add(v)
        topo(self)
        self.grad = 1.0
        rtopo = ls[::-1]
        for val in rtopo:
            val._backward()

    def __repr__(self):
        return f"Value(data={self.data})"


"""
x1 = Value(1)
w1 = Value(0.5)
x2 = Value(-2)
w2 = Value(0)
b = Value(-0.1137)

x1w1 = x1 * w1
x2w2 = x2 * w2
s = x1w1 + x2w2
n = s + b
o = n.tanh()
o.grad = 1.0

o.backward()

print(n.grad)
print(s.grad, b.grad)
print(x1w1.grad, x2w2.grad)
print(x1.grad, w1.grad)
print(x2.grad, w2.grad)


a = Value(2.0); b = Value(-1.0); d = Value(3.0)
c = a + b
g = c * d
f = g + c
f.backward()

print(a.grad)

"""

n = Value(0.8)
o = n.tanh()

o.backward()
print(o, n.grad)



"""

o._backward()
print(n.grad)

n._backward()
print(s.grad, b.grad)
s._backward()
print(x1w1.grad, x2w2.grad)
x1w1._backward()
print(x1.grad, w1.grad)
x2w2._backward()
print(x2.grad, w2.grad)

print(x1w1._prev)


print(n, o.data)

x1.grad = (1 - math.tanh(n)**2) * 1 * 1 * w1.data
w1.grad = (1 - math.tanh(n)**2) * 1 * 1 * x1.data
x2.grad = (1 - math.tanh(n)**2) * 1 * 1 * w2.data
w2.grad = (1 - math.tanh(n)**2) * 1 * 1 * x2.data
b.grad = (1 - math.tanh(n)**2) * 1

o.grad = 1
n1.grad = 1 - math.tanh(n)**2
x1w1.grad = (1 - math.tanh(n)**2) * 1 * 1
x2w2.grad = (1 - math.tanh(n)**2) * 1 * 1

print("my predcited value:", w1.grad)

print(n1._backward)
print(s._backward)

h = 0.001
x1 = Value(2)
x2 = Value(0)
w1 = Value(-3 + h)
w2 = Value(1)
b = Value(6.8813735870195432)

x1w1 = x1 * w1
x2w2 = x2 * w2
s = x1w1 + x2w2
n = s + b
o2 = n.tanh()
n = n.data
print(n, o.data)

print("do/dw1", (o2.data - o.data) / h)



a = Value(3.0)
b = Value(-4.0)
c = Value(0.5)
e = a + b
d = e * c
f = Value(-3)
L = d * f

print(f"{L.data}")

h = 0.001
a = Value(3.0)
b = Value(-4.0)
c = Value(0.5 + h)
e = a + b
d = e * c
f = Value(-3)
L2 = d * f

res = (L2.data - L.data)
fin = res / h

print(f"{fin}")



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
"""