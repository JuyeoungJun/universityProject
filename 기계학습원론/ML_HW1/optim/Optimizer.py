import numpy as np

# DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!! 

class SGD:
    def __init__(self):
        pass


    def update(self, w, grad, lr):
        """
        [Inputs]
            w : current weight
            grad : gradient for w
            lr : learning rate

        [Outputs]
            updated_weight : updated weight.
        """
        updated_weight=None
        # ========================= EDIT HERE =========================
        updated_weight = 0
        updated_weight = w - grad*lr         
        # =============================================================
        return updated_weight
