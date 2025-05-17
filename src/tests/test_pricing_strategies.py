import unittest
import numpy as np
from src.strategies.avellaneda_stoikov_abm import AvellanedaStoikovStrategyABM
from strategies.symmetric_strategy import SymmetricStrategy


class TestPricingStrategies(unittest.TestCase):
    def setUp(self):
        self.current_price = 100.0
        self.inventory = 10
        self.time_remaining = 1.0
        self.gamma = 0.1
        self.sigma = 2.0
        self.k = 1.5

        self.astoikov = AvellanedaStoikovStrategyABM(
            gamma=self.gamma,
            sigma=self.sigma,
            k=self.k
        )

        self.symmetric = SymmetricStrategy(
            gamma=self.gamma,
            sigma=self.sigma,
            k=self.k
        )

    def test_avellaneda_reservation_price(self):
        expected_price = self.current_price - self.inventory * self.gamma * (self.sigma ** 2) * self.time_remaining
        calculated = self.astoikov.calculate_reservation_price(self.current_price, self.inventory, self.time_remaining)
        self.assertAlmostEqual(calculated, expected_price, places=5)

    def test_avellaneda_spread(self):
        spread = self.gamma * (self.sigma ** 2) * self.time_remaining + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        bid_spread, ask_spread = self.astoikov.calculate_spread(self.current_price, self.inventory, self.time_remaining)
        self.assertAlmostEqual(bid_spread + ask_spread, spread, places=5)

    def test_symmetric_reservation_price(self):
        calculated = self.symmetric.calculate_reservation_price(self.current_price, self.inventory, self.time_remaining)
        self.assertEqual(calculated, self.current_price)

    def test_symmetric_spread(self):
        spread = self.gamma * (self.sigma ** 2) * self.time_remaining + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        bid_spread, ask_spread = self.symmetric.calculate_spread(self.current_price, self.inventory, self.time_remaining)
        self.assertAlmostEqual(bid_spread + ask_spread, spread, places=5)


if __name__ == "__main__":
    unittest.main()
