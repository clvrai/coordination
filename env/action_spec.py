from collections import OrderedDict

import numpy as np

from util.logger import logger


class ActionSpec(object):
    def __init__(self, size, minimum=-1., maximum=1.):
        self.size = size
        if size == 0:
            self.shape = OrderedDict()
            self.dtype = OrderedDict()
        else:
            self.shape = OrderedDict([('default', size)])
            self.dtype = OrderedDict([('default', np.float)])
        shape = (size, )

        try:
            np.broadcast_to(minimum, shape=shape)
        except ValueError as numpy_exception:
            raise ValueError('minimum is not compatible with shape. '
                             'Message: {!r}.'.format(numpy_exception))

        try:
            np.broadcast_to(maximum, shape=shape)
        except ValueError as numpy_exception:
            raise ValueError('maximum is not compatible with shape. '
                            'Message: {!r}.'.format(numpy_exception))

        self._minimum = np.array(minimum)
        self._minimum.setflags(write=False)

        self._maximum = np.array(maximum)
        self._maximum.setflags(write=False)

    @property
    def minimum(self):
        return self._minimum

    @property
    def maximum(self):
        return self._maximum

    def keys(self):
        return self.shape.keys()

    def is_continuous(self, key):
        return self.dtype[key] == 'continuous'

    def __repr__(self):
        template = ('ActionSpec(shape={}, dtype={}, '
                    'minimum={}, maximum={})')
        return template.format(self.shape, self.dtype,
                               self._minimum, self._maximum)

    def __eq__(self, other):
        if not isinstance(other, ActionSpec):
            return False
        return (self.minimum == other.minimum).all() and (self.maximum == other.maximum).all()

    def sample(self):
        return np.random.uniform(low=self.minimum,
                                 high=self.maximum,
                                 size=self.size)

    def decompose(self, shape):
        assert isinstance(shape, OrderedDict)
        #assert self.size == sum(shape.values())
        if self.size != sum(shape.values()):
            logger.error("Check the action space (urdf: %d, shape: %d (%s))",
                         self.size, sum(shape.values()), shape)
        self.size = sum(shape.values())
        self.shape = shape
        self.dtype = OrderedDict([(k, "continuous") for k in shape.keys()])

    def add(self, key, dtype, size, minimum, maxmimum):
        self.size += size
        self.shape[key] = size
        self.dtype[key] = dtype

