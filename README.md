# ml-from-scratch

Neural networks built from first principles, step by step.

## Learning log

### Week 1 (Nov 2–8, 2026): micrograd engine

**What I built**
I built a Value class, which supports the following operators: +, *, -, pow / **, /, e^x and tanh, working in all different kinds of configurations (with fallback functions in case of reverse order or a variable that isn't a Value class). 
The backward() function sets self.grad = 1, applies a topological sort to the neuron's nodes and leafs and then calls _backward() in reverse order for every node, setting the gradients.

**Why it works**
The _backward() function of the nodes / operators assigns values to the gradients to the children of the operator. It works by applying the chain rule, multiplying the derivative of the parent function by the local derivative in respect to the variable. The backward() function needs a topological sort, because only when the gradients of all parent nodes have been calculated, can the children nodes' and leafs' gradients be calculated. The _backward() function uses += instead of =, because the gradients of nodes and leaves can be changed several times, in case they are used several times in other operations.

**Bugs and how I found them**
1. The visited-Bug: Because I didn’t add the nodes to a visited set of already visited nodes, it added them to the list several times, upscaling the gradients incorrectly. I found it by predicting the values analytically and comparing them to my output values, seeing the error and finding the fix by adding the visited list.
2. The __pow__-Bug: Instead of applying the power rule in the _backward() function, I exponentiated out.grad by other.data, which isn't how it works. Because I didn't predict the actual value mathematically, the output result didn't appear suspicious. Only the prediction for the easiest case exposed the mistake, such that I corrected the method, applying the power rule instead of raising out.grad to a power and also removed other.grad +=, because the exponent is a constant, not a parameter we need a gradient for.

**Next**
Next week I am planning to compare my engine with PyTorch and dive a bit into Multi-Layer-Perceptrons. I'm already very excited!