import unittest
from functools import partial

import jax
import jax.numpy as jnp
from random import randint
import augmax
from augmax.geometric_utils import CoordinateTracer

def mk_rand(*shape):
  key = jax.random.PRNGKey(randint(0, 1024 * 1024))
  return jax.random.normal(key, shape)

allclose = partial(jnp.allclose, rtol=1e-3, atol=1e-1)


class TestBasicOps(unittest.TestCase):
  def setUp(self):
    self.coords = mk_rand(2, 10, 10)

  def test_addition(self):
    for _ in range(10):
      coords = CoordinateTracer(self.coords)

      coords = coords + mk_rand(2)
      coords = mk_rand(2) + coords

      naive = coords._naive_evaluate()
      optim = coords.evaluate()

      self.assertTrue(allclose(naive, optim))

    def test_linear_perspective(self):
      for _ in range(10):
        coords = CoordinateTracer(self.coords)

        coords = coords @ mk_rand(2, 2)
        coords = coords @ mk_rand(3, 3)
        coords = mk_rand(2, 2) @ coords
        coords = mk_rand(3, 3) @ coords

        naive = coords._naive_evaluate(self.coords)
        optim = coords.evaluate(self.coords)

        self.assertTrue(allclose(naive, optim))

    def test_linear_and_addition(self):
      for _ in range(10):
        coords = CoordinateTracer(self.coords)

        coords = coords @ mk_rand(2, 2)
        coords = coords + mk_rand(2)
        coords = coords @ mk_rand(2, 2)
        coords = coords + mk_rand(2)

        naive = coords._naive_evaluate()
        optim = coords.evaluate()

        self.assertTrue(allclose(naive, optim))


if __name__ == '__main__':
  unittest.main()
