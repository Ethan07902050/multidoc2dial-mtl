import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from generative_qa import GenerativeQAModule

class AbsWeighting(GenerativeQAModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

    def get_share_params(self):
        return list(self.model.rag.question_encoder.parameters()) + list(self.model.rag.generator.model.encoder.parameters())

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.model.rag.question_encoder.zero_grad()
        self.model.rag.generator.model.encoder.zero_grad()

    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.
        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                self.manual_backward(self.rep, transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.manual_backward(self.rep[task], batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            self._reset_grad(new_grads)
    
    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    self.manual_backward(losses[tn], retain_graph=True) if (tn+1)!=self.task_num else self.manual_backward(losses[tn])
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    self.manual_backward(losses[tn], retain_graph=True) if (tn+1)!=self.task_num else self.manual_backward(losses[tn])
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads
    
    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
    
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.
        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.
        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads

class UW(GenerativeQAModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*self.task_num, device=self.device))
    
    def training_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)
        loss = (loss_tensors/(2*self.loss_scale.exp())+self.loss_scale/2).sum()
        weights = (1/(2*torch.exp(self.loss_scale))).detach().cpu().numpy()

        # Log weights and losses to tensorboard
        weight_log = {f'{task}_weight': w for task, w in zip(self.abb_names, weights)}
        loss_log = {f'{task}_loss': loss for task, loss in zip(self.abb_names, loss_tensors)}
        self.log_dict(weight_log, prog_bar=True)
        self.log_dict(loss_log, prog_bar=True)

        return loss

class EW(GenerativeQAModule):
    def combine_loss(self, loss_tensors):
        return torch.sum(loss_tensors)        

class GenerationOnly(GenerativeQAModule):
    def combine_loss(self, loss_tensors):
        return loss_tensors[1]

class Linear(GenerativeQAModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.weight = torch.tensor([1.0, 0.0])
        dataset_size = len(self.train_dataloader().dataset)
        total_steps = math.ceil(dataset_size // hparams.train_batch_size) * hparams.max_epochs
        self.delta = 1 / total_steps
    
    def combine_loss(self, loss_tensors):
        weight_log = {f'{task}_weight': w for task, w in zip(self.abb_names, self.weight)}
        self.log_dict(weight_log, prog_bar=True)
        loss = torch.dot(loss_tensors, self.weight.to(self.device))
        self.weight[0] -= self.delta
        self.weight[1] += self.delta
        return loss      


class GradNorm(AbsWeighting):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # For manual backward
        self.automatic_optimization = False 

        # Gradient accumulation
        self.gradient_accumulation_steps = 8
        self.accumulated_loss = [0] * self.task_num

        self.epoch = 0
        self.loss_scale = nn.Parameter(torch.tensor([1.0]*self.task_num, device=self.device))
        self.train_loss_buffer = np.zeros([self.task_num, hparams.max_epochs])
        
        # https://libmtl.readthedocs.io/en/latest/docs/user_guide/mtl.html
        self.rep_grad = False

    def gradnorm_backward(self, losses):
        alpha = 1.5
        if self.epoch >= 1:
            loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)
            grads = self._get_grads(losses, mode='backward')
            if self.rep_grad:
                per_grads, grads = grads[0], grads[1]
                
            G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
            G = G_per_loss.mean(0)
            L_i = torch.Tensor([losses[tn].item()/self.train_loss_buffer[tn, 0] for tn in range(self.task_num)]).to(self.device)
            r_i = L_i/L_i.mean()
            constant_term = (G*(r_i**alpha)).detach()
            L_grad = (G_per_loss-constant_term).abs().sum(0)
            self.manual_backward(L_grad)
            loss_weight = loss_scale.detach().clone()
            
            if self.rep_grad:
                self._backward_new_grads(loss_weight, per_grads=per_grads)
            else:
                self._backward_new_grads(loss_weight, grads=grads)
            return loss_weight.cpu().numpy()
        else:
            loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
            self.manual_backward(loss)
            return np.ones(self.task_num)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        loss_tensors = self._step(batch) # [grounding_loss, generation_loss]
        
        for i in range(self.task_num):
            loss_tensors[i] /= self.gradient_accumulation_steps
            self.accumulated_loss[i] += loss_tensors[i].item()

        # Log weights to tensorboard
        weights = self.gradnorm_backward(loss_tensors)
        self.log_dict({f'{task}_weight': w for task, w in zip(self.abb_names, weights)}, prog_bar=True)

        # Gradient accumulation
        # Ref: https://colab.research.google.com/github/kozodoi/website/blob/master/_notebooks/2021-02-19-gradient-accumulation.ipynb
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.trainer.train_dataloader):
            logs = {f'{name}_loss': loss for name, loss in zip(self.abb_names, self.accumulated_loss)}
            self.log_dict(logs, prog_bar=True)
            self.accumulated_loss = [0] * self.task_num
            
            # Gradient clipping
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.1)
            opt.step()
            opt.zero_grad()
            sch.step()

        return loss_tensors

    def training_epoch_end(self, outputs):
        losses = [x['loss'].cpu().detach().numpy() for x in outputs]
        self.train_loss_buffer[:, self.epoch] = np.mean(losses, axis=0)
        self.epoch += 1