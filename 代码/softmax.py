import numpy as np
def softmax(f):
    f -= np.max(f, axis=-1, keepdims=True)
    exp_f = np.exp(f)
    sum_exp_f = np.sum(exp_f, axis=-1, keepdims=True)
    return exp_f / sum_exp_f

# Example data (logits)
logits = np.array([[1.0, 2.0, 3.0],
                   [1.0, 2.0, 1.0],
                   [1.0, 1.0, 2.0]])

# Applying softmax to the example data
softmax_values = softmax(logits)
print(softmax_values)

# f -= np.max(f)：这一步是为了数值稳定性。通过减去数组中的最大值，避免在计算 exp 时产生数值溢出。
# np.exp(f)：计算 f 中每个元素的指数。
# np.sum(np.exp(f))：计算所有指数值的和。
# np.exp(f) / np.sum(np.exp(f))：每个指数值除以指数值的总和，确保结果是一个概率分布，各元素之和为 1。