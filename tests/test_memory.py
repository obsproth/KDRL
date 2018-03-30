from unittest import TestCase
from kdrl.memory import *
import numpy as np

class TestSingleActionMemory(TestCase):
    def test_discrete(self):
        capacity = 3
        state_shape = (3, 2)
        num_actions = 4
        memory = SingleActionMemory(capacity, state_shape)
        s = np.arange(state_shape[0] * state_shape[1]).reshape(state_shape) + 10
        memory.start_episode(s)
        memory.set_action(0)
        memory.step(s + 1, 101)
        memory.set_action(1)
        memory.step(s + 2, 102)
        memory.set_action(2)
        memory.step(s + 3, 103)
        memory.set_action(3)
        memory.end_episode(s + 4, 104)
        self.assertTrue(np.array_equal(memory.s[0], s + 3))
        self.assertTrue(np.array_equal(memory.s[1], s + 1))
        self.assertTrue(np.array_equal(memory.a, np.array([3, 1, 2])))
        self.assertTrue(np.array_equal(memory.s[0] + 1, memory.ns[0]))
        self.assertTrue(np.array_equal(memory.ns[1], s + 2))
        self.assertTrue(np.array_equal(memory.s[2] + 1, memory.ns[2]))
        self.assertTrue(np.array_equal(memory.r, np.array([104, 102, 103])))
        self.assertTrue(not memory.c[0])
        self.assertTrue(memory.c[1])
        self.assertTrue(memory.c[2])
        for i in range(capacity):
            for elem in memory.sample(i):
                self.assertEqual(len(elem), i)

