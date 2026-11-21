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
I also compared the functionality of engine.py to PyTorch, it was very similar, although pytorch doesn't store the grad values of the non-leaf nodes after running backward() on them, to safe memory, so we have to x.retain_grad(). But it *also* doesn't automatically store the grads of the leafs, also to store memory, so for them we have to set x.requires_grad = True.

### How it works
Nin of the next layer has to be Nout of the layer before, because the outputs of all the Neurons have to be considered by the Neurons in the next layer.
You can count the amount of parameters by adding up all parameters of each layer. The parameters of every layer can be described as amount of neurons * (amount weights + 1 bias). So for example for: MLP(3, [4, 4, 1]), it's 4 * (3 + 1) + 4 * (4 +1) + 1 * (4 + 1) = 41

### Training
To train the neural net the following steps have to be done:
1. determine a data set for the input values and for the desired output values
2. determine the loss in respect to the predicted value and desired values, performing a forward pass
3. set p.grad to 0 (because otherwise we would add up gradients from different values of the function, e. g. resulting in x.grad = 3 * previous g after three runs, making the step size significantly bigger every time) and perform a backward pass
4. Change p.data in the opposite direction to the loss function (so -lr to *minimize* loss), with a well thought through step size
(5. iterate a bunch of times)
If the lr is too big, we oscillate, if it's too small we don't get close enough fast enough to a small loss.

### Bug / surprise
I counted the amount of p.grad = 0.0 with a dataset of "saturated" neurons, before actually doing the backward pass, setting all parameters to be 0.0, because they were initiatlly set to 0.0 and if I never do a backward pass they stay that way. After actually doing the backward pass, only when the first layer was saturated (meaning that math.tanh(n), due to the hight input values, outputs 1.0 as the closest representable float), the first layer (16 parameters) were 0.0, so resistant to learning. It also taught me that one very big data entry, out of the ordinary, could freeze up a part / example of the data set, decreasing the amount of times a satisfying amount of loss is reached after several runs.
The grad of a parameter is the *sum* of the contributions of all examples, so all the contributions have to be 0.0 themselves, so every example frozen up, for the parameter's grad to actually be 0.0.

### Next
makemore for medical texts

## Week 3 (16–20 Nov)

### What I built and why it works
I started and finished building makemore 1 myself. `part1.py` generates names from the probability distribution of bigrams, which is calculated directly from the counts, independent of any loss function.

`part2.py` contains a one-layer "neural net": the one-hot encoded inputs `xs` are multiplied by the weight matrix `W` (randomly initialized), the resulting logits are exponentiated and interpreted as counts, normalized into probabilities, and the negative log likelihood is calculated as the loss. Calling `backward()` on the loss gives the gradients, and the weights are updated by subtracting the learning rate (which is relatively big) times the gradients. This training loop is repeated until the loss gets close to the loss of the counting approach. It cannot go below it: the counting frequencies are already optimal on the training data (maximum likelihood), and the net cannot represent anything more than that.

Both approaches use smoothing: the counting approach adds fake counts to `N`, the neural net adds a regularization term `(W**2).mean()` to the loss, which pulls `W` towards zero and acts like smoothing.

### How I tested my explanation
In Week 2, one of my training examples (10^12) often kept the loss above 0.1, and I explained it with the first layer being frozen for that example (tanh saturates, the gradient is exactly 0). To test this, I replaced 10^12 with 0.5 and trained both versions from 20 random starts each (learning rate 0.05, 50 steps, success = loss < 0.1). I predicted 20/20 for 0.5 and got 20/20, compared to 16/20 with 10^12. One successful run would have proven nothing, since about 2/3 of all runs succeed anyway; a check only counts if the opposite hypothesis would predict a different result.

### Bugs I encountered along the way
**Broadcasting error:** While normalizing the bigram counts, I forgot `keepdim=True` in `N.sum(1)`. The row sums were then broadcast as a row instead of a column, so every column was divided by the wrong sum, without any error message. The check `P.sum() == 27` could not reveal it, because in `N` the row sums equal the column sums: every character has exactly one predecessor and one successor, since `.` starts and ends every name. Checking a single row (`P[0].sum()`) would have shown it.

**Generator trap:** After training, `part2.py` generated different names than `part1.py`, even though the losses were very close and I used the same seed. I had already used the generator to initialize `W`, so it produced different random numbers. Re-creating the generator with the same seed right before sampling fixed it.

**Row-sum errors:** A uniform distribution built from ones had row sums of 27 instead of 1, so every probability was 1 and the loss was log(1) = 0, which looked perfect but was just an invalid distribution. Later, I added the smoothing counts only to the numerator, which made the training loss seem to drop. Lesson: after building any `P`, check `P.sum(1)` immediately; a training loss of 0, or one below the counting model, is always a bug.