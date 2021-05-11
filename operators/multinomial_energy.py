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

import nifty6 as ift
import numpy as np

class CategoricalEnergy(ift.EnergyOperator):
    """
    The negative logarithm of the categorical distribution for outcomes d as a function
     of the classification probabilities.

    Parameters
    ----------
    d : Nifty-Field of positive integers
    The outcomes of the multinomial experiments.
    scale : positive float
    The scaling factor used to weight the impact of this likelihood.
    """
    def __init__(self, d, scale=1.):
        if not isinstance(d, ift.Field) or not np.issubdtype(d.dtype, np.integer):
            raise TypeError
        if not np.all(np.logical_or(d.val== 0, d.val == 1)):
            raise ValueError
        self._d = d
        self._domain = ift.DomainTuple.make(d.domain)
        self._scale = scale

    def apply(self, x):
        self._check_input(x)
        v = -x.log().vdot(self._d) * self._scale
        if not isinstance(x, ift.Linearization):
            return v
        if not x.want_metric:
            return v
        met = ift.makeOp(self._scale/(x.val))
        met = ift.SandwichOperator.make(x.jac, met)
        return v.add_metric(met)
