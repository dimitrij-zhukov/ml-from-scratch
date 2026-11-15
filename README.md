# ml-from-scratch

Neural networks built from first principles, step by step.

## Learning log

### Week 1 (Nov 2–8, 2026): micrograd engine

**What I built**
I built a Value class, which supports the following operators: +, *, -, pow / **, /, e^x and tanh, working in all different kinds of configurations (with fallback functions in case of reverse order or a variable that isn't a Value class). 
The backward() function sets self.grad = 1, applies a topological sort to the neuron's nodes and leafs and then calls _backward() in reverse order for every node, setting the gradients.
_backward() is the chain rule, so local derivative * out.grad, chaining (starting from the final equation) all of the derivaties together to get the nodes' or leafs' grads.

**Why it works**
The _backward() function of the nodes / operators assigns values to the gradients to the children of the operator. It works by applying the chain rule, multiplying the derivative of the parent function by the local derivative in respect to the variable. The backward() function needs a topological sort, because only when the gradients of all parent nodes have been calculated, can the children nodes' and leafs' gradients be calculated. The _backward() function uses += instead of =, because the gradients of nodes and leaves can be changed several times, in case they are used several times in other operations.

**Bugs and how I found them**
1. The visited-Bug: Because I didn’t add the nodes to a visited set of already visited nodes, it added them to the list several times, upscaling the gradients incorrectly. I found it by predicting the values analytically and comparing them to my output values, seeing the error and finding the fix by adding the visited list.
2. The __pow__-Bug: Instead of applying the power rule in the _backward() function, I exponentiated out.grad by other.data, which isn't how it works. Because I didn't predict the actual value mathematically, the output result didn't appear suspicious. Only the prediction for the easiest case exposed the mistake, such that I corrected the method, applying the power rule instead of raising out.grad to a power and also removed other.grad +=, because the exponent is a constant, not a parameter we need a gradient for.

**Next**
Next week I am planning to compare my engine with PyTorch and dive a bit into Multi-Layer-Perceptrons. I'm already very excited!

## Week 2 (Nov 13–15)

### What I built
I built the whole nn.py consisting of the 
1. Neuron Class: Made out of randoms weights (paired with the input values) and one random bias, meaning n = w1 * x1 + w2 * x2 + ... + b. It outputs a value of tanh(n) and return it's parameters.
2. Layer Class: Made out of a certain (nout) amount of Neurons, returns a list (or singular value if it's only 1 Neuron) of the Neurons outputs. It also can return the parameters of all the Neurons it consists of.
3. MLP Class: Made out of several (len(nouts)) Layer classes, it iterates through the layers, inputing the outputs of the last layer into the next once called, return the last layers value(s). It also can return the parameters of all the Layers it consists of, so also out of all the Neurons.
I also compared the functionality of engine.py to PyTorch, it was very similar, through pytorch doesn't automatically determine the gradient of leafs.

### How it works
Nin of the next layer has to be Nout of the layer before, because the outputs of all the Neurons have to be considered by the Neurons in the next layer. I don't know whether that has to always hold true, but for such simple MLPs it makes great sense.
You can count the amount of parameters by adding up all parameters of each layer. The parameters of every layer can be described as amount of neurons * (amount weights + 1 bias). So for example for: MLP(3, [4, 4, 1]), it's 4 * (3 + 1) + (4 * 4 +1) + 1 * (4 + 1) = 41

### Training
To train the neural net the following steps have to be done:
1. determine a data set for the input values and for the desired output values
2. determine the loss in respect to the predicted value and desired values, performing a forward pass
3. set p.grad to 0 (because otherwise you could very well overshoot, such that the value oscillates between negative and positive) and perform a backward pass
4. Change p.data in the opposite direction to the loss function (so -lr to *minimize* loss), with a well thought through step size
(5. iterate a bunch of times)
If the lr is too big, we oscillate, if it's too small we don't get close enough fast enough to a small loss.

### Bug / surprise
I counted the amount of p.grad = 0.0 with a dataset of "saturated" neurons, before actually doing the backward pass, setting all parameters to be 0.0. After actually doing the backward pass, only if and if all neurons were saturated, the first layer (16 parameters) were 0.0, so resistant to learning. It also taught me that one very big data entry, out of the ordinary, could freeze up a whole layer, complicating the training process very significantly.

### Next
makemore for medical texts