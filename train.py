
from predictor import PointWiseEncoder
import torch

model = PointWiseEncoder()
torch.save(model.state_dict(), 'model.pkl')