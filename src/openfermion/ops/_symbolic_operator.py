#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""SymbolicOperator is a base class for QubitOperator and FermionOperator"""
import copy
import itertools

import numpy


EQ_TOLERANCE = 1e-12



class SymbolicOperatorError(Exception):
    pass


class SymbolicOperator(object):
    """
    """

    @classmethod
    def zero(cls):
        """
        Returns:
            additive_identity (SymbolicOperator):
                A symbolic operator o with the property that o+x = x+o = x for
                all fermion operators x.
        """
        # Maybe throw a TypeError if called using SymbolicOperator? Or maybe 
        # just don't expose SymbolicOperator to users
        
        return cls(term=None)

    @classmethod
    def identity(cls):
        """
        Returns:
            multiplicative_identity (FermionOperator):
                A symbolic operator u with the property that u*x = x*u = x for
                all fermion operators x.
        """
        return cls(term=())

    def compress(self, abs_tol=EQ_TOLERANCE):
        """
        Eliminates all terms with coefficients close to zero and removes
        imaginary parts of coefficients that are close to zero.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        new_terms = {}
        for term in self.terms:
            coeff = self.terms[term]
            if abs(coeff.imag) <= abs_tol:
                coeff = coeff.real
            if abs(coeff) > abs_tol:
                new_terms[term] = coeff
        self.terms = new_terms

    def isclose(self, other, rel_tol=EQ_TOLERANCE, abs_tol=EQ_TOLERANCE):
        """
        Returns True if other (QubitOperator) is close to self.

        Comparison is done for each term individually. Return True
        if the difference between each term in self and other is
        less than the relative tolerance w.r.t. either other or self
        (symmetric test) or if the difference is less than the absolute
        tolerance.

        Args:
            other(QubitOperator): QubitOperator to compare against.
            rel_tol(float): Relative tolerance, must be greater than 0.0
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        # terms which are in both:
        for term in set(self.terms).intersection(set(other.terms)):
            a = self.terms[term]
            b = other.terms[term]
            # math.isclose does this in Python >=3.5
            if not abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol):
                return False
        # terms only in one (compare to 0.0 so only abs_tol)
        for term in set(self.terms).symmetric_difference(set(other.terms)):
            if term in self.terms:
                if not abs(self.terms[term]) <= abs_tol:
                    return False
            elif not abs(other.terms[term]) <= abs_tol:
                return False
        return True

    def __rmul__(self, multiplier):
        """
        Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for  SymbolicOperator and left multiply
        is also queried as the default behavior.

        Args:
          multiplier: A scalar to multiply by.

        Returns:
          product: A new instance of SymbolicOperator.

        Raises:
          TypeError: Object of invalid type cannot multiply SymbolicOperator.
        """
        if not isinstance(multiplier, (int, float, complex)):
            raise TypeError(
                'Object of invalid type cannot multiply with ' 
                + type(self) + '.')
        return self * multiplier

    def __truediv__(self, divisor):
        """
        Return self / divisor for a scalar.

        Note:
            This is always floating point division.

        Args:
          divisor: A scalar to divide by.

        Returns:
          A new instance of SymbolicOperator.

        Raises:
          TypeError: Cannot divide local operator by non-scalar type.

        """
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide ' + type(self) 
                    + ' by non-scalar type.')
        return self * (1.0 / divisor)

    def __div__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__truediv__(divisor)

    def __itruediv__(self, divisor):
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide ' + type(self) 
                    + ' by non-scalar type.')
        self *= (1.0 / divisor)
        return self

    def __idiv__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__itruediv__(divisor)

    def __neg__(self):
        """
        Returns:
            negation (SymbolicOperator)
        """
        return -1 * self
