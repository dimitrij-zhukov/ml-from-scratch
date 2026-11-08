# ml-from-scratch

Neural networks built from first principles, step by step.

## Learning log

### Week 1 (Nov 2–8, 2026): micrograd engine

**What I built**
I built a Value class, which supports the following operators: +, *, -, pow / **, e^x and tanh, working in all different kinds of configurations (with fallback functions in case of reverse order or a variable that isn't a Value class). 
The Value class also has integrated backpropagation in case the backward() function (topologically sorted) works backward from the node it is applied on, iterating through the nodes, defining their gradients, until it reaches the leaves and defines their gradients, updating everything in case it gets affected several times.

**Why it works**
The _backward() function of the nodes / operators assigns values to the gradients to the children of the operator. The backward() function needs a topological sort, because only in that case it can iterate through all nodes truly backwards, correctly. The _backward() function uses += instead of =, because the gradients of nodes and leaves can be changed several times, in case they are used several times in other operations.

**Bugs and how I found them**
1. The visited-Bug: Because I didn't remove already visited nodes to the pool of already visited nodes, it added them to the list several times, upscaling the gradients uncorrectly. I found it by predicting the values analytically and comparing them to my outputed values, seeing the error and finding the fix by adding the visited list.
2. The __pow__-Bug: Instead of applying the power rule in the _backward() function, I exponentiated self.grad by other.data, which isn't how it works. The results deviated from the predictions (which I mistakenly didn't do, because I was lazy), but in the end I found the bug, applied the power rule and also remove the other.grad +=, because only self.grad is an actual gradient.

**Next**
Next week I am planning to implement PyTorch capabilities and dive a bit into Multi-Layer-Perceptrons. I'm already very excited!