# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2020 Max-Planck-Society
# Author: Jakob Knollmueller

import nifty6 as ift
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TensorFlowOperator(ift.Operator):
    """
    A wrapper for TensorFlow tensors as Nifty operators.
    The Jacobian and its adjoint are calculated via TensorFlow auto-differentiaiton.

    Parameters
    ----------
    tf_op : TensorFlow Tensor
        The tensor corresponding to the output layer.
    argument : TensorFlow Tensor
        The input tensor.
    domain : Nifty Domain
        The input-domain of the operator.
    target : Nifty Domain
        The output-domain of the operator.
    add_domain_axis : boolean
        Wheter to add an axis to the input to match the tensor shape. (default: False)
    add_target_axis : boolean
        Wheter to add an axis to the output to match the domain shape. (default: False)
    """

    def __init__(self, tf_op, argument, domain, target,
                 add_domain_axis=False, add_target_axis=False):
        self._target = ift.DomainTuple.make(target)
        self._domain = ift.DomainTuple.make(domain)
        self._tf_op = tf_op
        self._argument = argument
        if add_target_axis:
            self._output_shape = (1,) + self._target.shape + (1,)
        else:
            self._output_shape = (1,) + self._target.shape
        if add_domain_axis:
            self._input_shape = (1,) + self._domain.shape + (1,)
        else:
            self._input_shape = (1,) + self._domain.shape

        self._d_x = tf.placeholder(tf.float32,  self._input_shape)
        self._d_y = tf.placeholder(tf.float32,  self._output_shape)
        self._adjoint_jac = self.adjoint_jacobian(tf_op, self._argument, self._d_y)
        self._jac = self.jacobian(tf_op, self._argument, self._d_x)

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        val = x.val.val if lin else x.val
        val = val.reshape(self._input_shape)
        res = self._tf_op.eval(feed_dict={self._argument: val}).squeeze()
        res = ift.makeField(self._target, res)
        if lin:
            _jac = TensorflowJacobian(self._jac, self._adjoint_jac, val,
                                      self._argument, self._d_x, self._d_y,
                                      self._domain, self._target, self._input_shape,
                                      self._output_shape)
            jac = _jac(x.jac)
            return x.new(res, jac)
        return res

    def jacobian(self, y, x, d_x):
        z = tf.zeros_like(y)
        g = tf.gradients(y, x, grad_ys=z)
        return tf.gradients(g, z, grad_ys=d_x)[0]

    def adjoint_jacobian(self, y, x, d_y):
        return tf.gradients(y, x, grad_ys=d_y)[0]


class TensorflowJacobian(ift.LinearOperator):
    """
    The Jacobian of a TensorFlowOperator as linear Nifty operator.

    Parameters
    ----------
    jac : TensorFlow Tensor
        The Jacobian of the TensorFlow tensor w.r.t. the input.
    adjoint_jac : TensorFlow Tensor
        The adjoint Jacobian of the TensorFlow tensor w.r.t. the input.
    loc : Nifty Field
        The location at which the Jacobian is evaluated.
    argument : Nifty domain
        The input of the original tensor.
    d_x : TensorFlow Tensor
        The input tensor for the Jacobian.
    d_y : TensorFlow Tensor
        The input tensor for the adjoint Jacobian.
    domain : Nifty Domain
        The input-domain of the operator.
    target : Nifty Domain
        The output-domain of the operator.
    input_shape : tuple
        The shape of the input.
    output_shape : tuple
        The shape of the output.
    """
    def __init__(self, jac, adjoint_jac, loc, argument, d_x, d_y, domain,
                 target, input_shape, output_shape):
        self._target = ift.DomainTuple.make(target)
        self._domain = ift.DomainTuple.make(domain)
        self._output_shape = output_shape
        self._input_shape = input_shape

        self._jac = jac
        self._adjoint_jac = adjoint_jac
        self._argument = argument
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._loc = loc
        self._d_x = d_x
        self._d_y = d_y

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            x = x.reshape(self._input_shape)
            res = self._jac.eval(feed_dict={self._d_x: x, self._argument: self._loc})
            return ift.makeField(self.target, res.squeeze())
        x = x.reshape(self._output_shape)
        res = self._adjoint_jac.eval(feed_dict={self._d_y: x, self._argument: self._loc})
        return ift.makeField(self.domain, res.squeeze())
