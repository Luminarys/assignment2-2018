from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) * B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(0., dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i), B))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: tvm.select(A(*i) < 0, 0.0, B(*i)))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")

    if transposeA and transposeB:
        k = tvm.reduce_axis((0, shapeA[0]))
        C = tvm.compute((shapeA[1], shapeB[0]), lambda x, y: tvm.sum(A[k, x] * B[y, k], axis=k), name="C")
    elif transposeA:
        k = tvm.reduce_axis((0, shapeA[0]))
        C = tvm.compute((shapeA[1], shapeB[1]), lambda x, y: tvm.sum(A[k, x] * B[k, y], axis=k), name="C")
    elif transposeB:
        k = tvm.reduce_axis((0, shapeA[1]))
        C = tvm.compute((shapeA[0], shapeB[0]), lambda x, y: tvm.sum(A[x, k] * B[y, k], axis=k), name="C")
    else:
        k = tvm.reduce_axis((0, shapeA[1]))
        C = tvm.compute((shapeA[0], shapeB[1]), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name="C")

    s = tvm.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=32)
    s[C].reorder(xo, yo, xi, yi)
    s[C].vectorize(yi)
    #print(tvm.lower(s, [A, B], simple_mode=True))
    raise Exception("")
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF
    """For a challenge, treat the general case for stride and padding."""
    in_size = (H, W)
    kern_size = (R, S)
    out_size = (in_size[0] - kern_size[0] + 1, in_size[1] - kern_size[1] + 1)
    batch = N
    filters = M

    A = tvm.placeholder(shapeX, dtype=dtype, name="A")
    B = tvm.placeholder(shapeF, dtype=dtype, name="B")
    rc = tvm.reduce_axis((0, C), name="rc")
    rx = tvm.reduce_axis((0, kern_size[0]), name="rx")
    ry = tvm.reduce_axis((0, kern_size[1]), name="ry")

    C = tvm.compute((batch, filters, out_size[0], out_size[1]), lambda n, f, x, y: tvm.sum(
        A[n, rc, x + rx, y + ry] * B[f, rc, rx, ry],
        axis=[rx, ry, rc]), name="C")
    s = tvm.create_schedule(C.op)
    raise Exception("")
    return tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.compute(shape, )
    C = tvm.compute(shape, lambda *i: B(*i) / tvm.sum(B))

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
