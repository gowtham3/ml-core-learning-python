def forward_multiply_gate(a, b):
    return a * b


def forward_add_gate(a, b):
    return a + b


def forward_circuit(x, y, z):
    q = forward_add_gate(x, y)
    z = forward_multiply_gate(q, z)

    return z


x = -2
y = 5
z = -4
f = forward_circuit(x, y, z)
print("initial value: ", f)
# -12

q = forward_add_gate(x, y)

f = forward_multiply_gate(q, z)

derivative_f_wrt_z = q
derivative_f_wrt_q = z


derivative_q_wrt_y = 1
derivative_q_wrt_x = 1


derivative_f_wrt_x = derivative_f_wrt_q * derivative_q_wrt_x
derivative_f_wrt_y = derivative_f_wrt_q * derivative_q_wrt_y

gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]


step_size = 0.01

x = x + step_size * derivative_f_wrt_x
y = y + step_size * derivative_f_wrt_y
z = z + step_size * derivative_f_wrt_z


print("After one step: ", forward_circuit(x, y, z))
# -11.5924, better


