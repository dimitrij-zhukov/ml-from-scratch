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