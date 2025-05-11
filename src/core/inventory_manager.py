# core/inventory_manager.py
class InventoryManager:
    def __init__(self, initial_cash: float, initial_inventory: int):
        self.cash = initial_cash
        self.inventory = initial_inventory

    def update(self, delta_inventory: int, delta_cash: float):
        self.inventory += delta_inventory
        self.cash += delta_cash

