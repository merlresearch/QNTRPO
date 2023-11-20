# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import random
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "mask", "next_state", "reward"))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):

        self.memory.append(Transition(*args))

    def sample(self):

        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
