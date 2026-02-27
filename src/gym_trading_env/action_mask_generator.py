"""
Action Mask Generator for MTF Trading Gym Environment.

Generates boolean masks to prevent invalid trading actions based on pattern state.
Implements masking logic for H1 pattern-based trading with M15 trigger confirmation.
"""

import numpy as np


class ActionMaskGenerator:
    """
    Generates action masks for RL agent based on pattern state.

    Action masking ensures the RL agent can only select valid actions based on:
    - H1 pattern detection state (long/short windows)
    - M15 trigger confirmation
    - Current position state

    This prevents the agent from learning impossible or invalid action sequences.
    """

    # Action space constants
    ACTION_HOLD = 0
    ACTION_ENTER_LONG = 1
    ACTION_ENTER_SHORT = 2
    ACTION_CLOSE_POSITION = 3

    def __init__(self):
        """
        Initialize action mask generator.

        Sets up the action space size for the 4 possible actions:
        - Hold (do nothing)
        - Enter Long position
        - Enter Short position
        - Close existing position
        """
        self.action_space_size = 4

    def generate_mask(
        self,
        window_long: bool,
        window_short: bool,
        position_exists: bool,
        m15_trigger_valid: bool,
    ) -> np.ndarray:
        """
        Generate boolean action mask based on current pattern and position state.

        Masking rules:
        - Hold (0): ALWAYS True - agent can always choose to do nothing
        - EnterLong (1): True ONLY if window_long=True AND m15_trigger_valid=True
        - EnterShort (2): True ONLY if window_short=True AND m15_trigger_valid=True
        - ClosePosition (3): True ONLY if position_exists=True

        Args:
            window_long: True if H1 pattern opened long window
            window_short: True if H1 pattern opened short window
            position_exists: True if agent has open position
            m15_trigger_valid: True if M15 trigger detected in window

        Returns:
            Boolean numpy array of shape (4,) with mask for each action:
            [hold, enter_long, enter_short, close_position]
        """
        # Initialize mask - all False by default
        mask = np.zeros(self.action_space_size, dtype=bool)

        # Hold is ALWAYS valid - agent can always choose to do nothing
        mask[self.ACTION_HOLD] = True

        # EnterLong: requires both long window AND valid trigger
        mask[self.ACTION_ENTER_LONG] = window_long and m15_trigger_valid

        # EnterShort: requires both short window AND valid trigger
        mask[self.ACTION_ENTER_SHORT] = window_short and m15_trigger_valid

        # ClosePosition: only valid if position exists
        mask[self.ACTION_CLOSE_POSITION] = position_exists

        return mask
