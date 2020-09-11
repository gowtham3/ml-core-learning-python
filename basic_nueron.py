import math


# every Unit corresponds to a wire
class Unit:

    def __init__(self, value, grad):
        # Value computed in forward pass
        self.value = value

        # Value computed in backward pass
        self.grad = grad


class MultiplyGate:

    def __init__(self):
        # Storing input units and output units
        self.input1 = None
        self.input2 = None
        self.output = None

    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.output = Unit(input1.value * input2.value, 0)

        return self.output

    def backward(self):
        # Taking the gradient in the output unit and chain it with local gradient for multiply gates
        self.input1.grad += self.input2.value * self.output.grad
        self.input2.grad += self.input1.value * self.output.grad


class AddGate:

    def __init__(self):
        # Storing input units and output units
        self.input1 = None
        self.input2 = None
        self.output = None

    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.output = Unit(input1.value + input2.value, 0)

        return self.output

    def backward(self):
        # Add gate derivative wrt inputs is 1
        self.input1.grad += 1 * self.output.grad
        self.input2.grad += 1 * self.output.grad


class SigmoidGate:

    def __init__(self):
        self.gate_input = None
        self.output = None

    def forward(self, gate_input):
        self.gate_input = gate_input
        self.output = Unit(SigmoidGate.sigmoid(gate_input.value), 0)

        return self.output

    def backward(self):
        # Add gate derivative wrt inputs is 1
        sig_value = SigmoidGate.sigmoid(self.gate_input.value)
        self.gate_input.grad += sig_value * (1 - sig_value) * self.output.grad

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))


ax_gate = MultiplyGate()
by_gate = MultiplyGate()
ax_plus_by = AddGate()
ax_plus_by_plus_c = AddGate()
sigmoid_gate = SigmoidGate()


# Creating a nueron with f = sig(ax + by + c)
def forward_neuron(a, b, c, x, y):
    ax = ax_gate.forward(a, x)
    by = by_gate.forward(b, y)
    ax_p_by = ax_plus_by.forward(ax, by)
    ax_p_by_p_c = ax_plus_by_plus_c.forward(ax_p_by, c)

    return sigmoid_gate.forward(ax_p_by_p_c)


a = Unit(1, 0)
b = Unit(2, 0)
c = Unit(-3, 0)
x = Unit(-1, 0)
y = Unit(3, 0)

neuron = forward_neuron(a, b, c, x, y)

print("Forward neuron output: ", neuron.value)


# Backward propagation
neuron.grad = 1
sigmoid_gate.backward()
ax_plus_by_plus_c.backward()
ax_plus_by.backward()
by_gate.backward()
ax_gate.backward()


# Adjusting input values based on the gradient
step_size = 0.01
a.value += step_size * a.grad
b.value += step_size * b.grad
c.value += step_size * c.grad
x.value += step_size * x.grad
y.value += step_size * y.grad

output_after_one_round = forward_neuron(a, b, c, x, y)

print("Forward neuron output after one round: ", output_after_one_round.value)
