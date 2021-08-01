import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model,
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            alpha = self.optim_dict['learning_ratio']
            self.optimizer = optim.Adam(
                # [
                #     {'params': model.conv2d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.conv1d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.rnn.parameters()},
                #     {'params': model.classifier.parameters()},
                # ],
                # model.conv1d.fc.parameters(),
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)



# import pdb
# import torch
# import functools
# import numpy as np
# import torch.optim as optim
# from torch.nn.utils import clip_grad_norm_


# class Optimizer(object):
#     def __init__(self, model, optim_dict, decay_method="noam", max_grad_norm=0):
#         self._decay_step = 1
#         self._optim_dict = optim_dict
#         self._decay_method = decay_method
#         self._max_grad_norm = max_grad_norm
#         self._learning_rate = self._optim_dict['base_lr']
#         self._learning_rate_decay_fn = self.make_learning_rate_decay_fn()
#         if self._optim_dict["optimizer"] == 'SGD':
#             self.optimizer = optim.SGD(
#                 model,
#                 lr=self._learning_rate,
#                 momentum=0.9,
#                 nesterov=self._optim_dict['nesterov'],
#                 weight_decay=self._optim_dict['weight_decay']
#             )
#         elif self._optim_dict["optimizer"] == 'Adam':
#             self.optimizer = optim.Adam(
#                 model,
#                 lr=self._learning_rate,
#                 eps=1e-9,
#                 betas=self._optim_dict['betas'],
#                 weight_decay=self._optim_dict['weight_decay']
#             )
#         else:
#             raise ValueError()

#     def learning_rate(self):
#         """Returns the current learning rate."""
#         if self._learning_rate_decay_fn is None:
#             return self._learning_rate
#         scale = self._learning_rate_decay_fn(self._decay_step)
#         return scale * self._learning_rate

#     def zero_grad(self):
#         self.optimizer.zero_grad()

#     def step(self):
#         learning_rate = self.learning_rate()
#         for group in self.optimizer.param_groups:
#             group['lr'] = learning_rate
#             if self._max_grad_norm > 0:
#                 clip_grad_norm_(group['params'], self._max_grad_norm)
#         self.optimizer.step()
#         self._decay_step += 1

#     def state_dict(self):
#         return self.optimizer.state_dict()

#     def load_state_dict(self, state_dict):
#         self.optimizer.load_state_dict(state_dict)

#     def make_learning_rate_decay_fn(self):
#         """Returns the learning decay function from options."""
#         if self._decay_method == 'noam':
#             return functools.partial(
#                 self.noam_decay,
#                 warmup_steps=self._optim_dict['warmup_steps'],
#                 model_size=self._optim_dict['rnn_size']
#             )

#     def noam_decay(self, step, warmup_steps, model_size):
#         """Learning rate schedule described in
#         https://arxiv.org/pdf/1706.03762.pdf.
#         """
#         return (
#                 model_size ** (-0.5) *
#                 min(step ** (-0.5), step * warmup_steps ** (-1.5))
#         )
