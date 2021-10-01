import os
import sys

module = os.path.abspath(os.path.join(".."))
if module not in sys.path:
    sys.path.append(module)
    
from Chpt_14.Updated_Classes import Loss

import numpy as np

class MeanAbsoluteError_Loss(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs

        self.dinputs = self.dinputs / samples
        
class MeanSquaredError_Loss(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs

        self.dinputs = self.dinputs / samples