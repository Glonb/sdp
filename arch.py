import  torch
import  numpy as np
from    torch import optim, autograd


def concat(xs):
    """
    flatten all tensor from [d1,d2,...dn] to [d]
    and then concat all [d_1] to [d_1+d_2+d_3+...]
    """
    return torch.cat([x.view(-1) for x in xs])


class Arch:

    def __init__(self, model, args):
        self.momentum = args.momentum # momentum for optimizer of theta
        self.wd = args.wd # weight decay for optimizer of theta
        self.model = model # main model with respect to theta and alpha
        
        # this is the optimizer to optimize alpha parameter
        self.optimizer = optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_lr,
                                          betas=(0.8, 0.9),
                                          weight_decay=args.arch_wd)

    def comp_unrolled_model(self, x, trf, target, eta, optimizer):
        """
        loss on train set and then update w_pi, not-in-place
        :param optimizer: optimizer of theta, not optimizer of alpha
        """
        # forward to get loss
        loss = self.model.loss(x, trf, target)
        # flatten current weights
        theta = concat(self.model.parameters()).detach()
        # print('theta:', theta.shape)
        try:
            # fetch momentum data from theta optimizer
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)
        except:
            moment = torch.zeros_like(theta)

        # flatten all gradients
        dtheta = concat(autograd.grad(loss, self.model.parameters())).data
        
        # indeed, here we implement a simple SGD with momentum and weight decay
        # theta = theta - eta * (moment + weight decay + dtheta)
        theta = torch.sub(theta, moment + dtheta + self.wd * theta, alpha=eta)
        # construct a new model
        unrolled_model = self.construct_model_from_theta(theta)

        return unrolled_model

    def step(self, x_train, trf_train, target_train, x_valid, trf_valid, target_valid, eta, optimizer, unrolled):
        """
        update alpha parameter by manually computing the gradients
        :param optimizer: theta optimizer
        """
        # alpha optimizer
        self.optimizer.zero_grad()

        # compute the gradient and write it into tensor.grad
        # instead of generated by loss.backward()
        if unrolled:
            self.backward_step_unrolled(x_train, trf_train, target_train, x_valid, trf_valid, target_valid, eta, optimizer)
        else:
            # directly optimize alpha on w, instead of w_pi
            self.backward_step(x_valid, trf_valid, target_valid)

        self.optimizer.step()

    def backward_step(self, x_valid, trf_valid, target_valid):
        """
        simply train on validate set and backward
        """
        loss = self.model.loss(x_valid, trf_valid, target_valid)
        # both alpha and theta require grad but only alpha optimizer will
        # step in current phase.
        loss.backward()

    def backward_step_unrolled(self, x_train, trf_train, target_train, x_valid, trf_valid, target_valid, eta, optimizer):
        """
        train on validate set based on update w_pi
        :param eta: 0.01, according to author's comments
        :param optimizer: theta optimizer
        """

        # theta_pi = theta - lr * grad
        unrolled_model = self.comp_unrolled_model(x_train, trf_train, target_train, eta, optimizer)
        # calculate loss on theta_pi
        unrolled_loss = unrolled_model.loss(x_valid, trf_valid, target_valid)

        # this will update theta_pi model, but NOT theta model
        unrolled_loss.backward()
        # grad(L(w', a), a), part of Eq. 6
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self.hessian_vector_product(vector, x_train, trf_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            # g = g - eta * ig, from Eq. 6
            g.data.sub_(ig.data, alpha=eta)

        # write updated alpha into original model
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def construct_model_from_theta(self, theta):
        """
        construct a new model with initialized weight from theta
        it use .state_dict() and load_state_dict() instead of
        .parameters() + fill_()
        :param theta: flatten weights, need to reshape to original shape
        :return:
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = v.numel()
            # restore theta[] value to original shape
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def hessian_vector_product(self, vector, x, trf, target, r=1e-2):
        """
        slightly touch vector value to estimate the gradient with respect to alpha
        refer to Eq. 7 for more details.
        :param vector: gradient.data of parameters theta
        """
        R = r / concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            # w+ = w + R * v
            p.data.add_(v, alpha=R)
        loss = self.model.loss(x, trf, target)
        # gradient with respect to alpha
        grads_p = autograd.grad(loss, self.model.arch_parameters())


        for p, v in zip(self.model.parameters(), vector):
            # w- = (w+R*v) - 2R*v
            p.data.sub_(v, alpha=2 * R)
        loss = self.model.loss(x, trf, target)
        grads_n = autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # w = (w+R*v) - 2R*v + R*v
            p.data.add_(v, alpha=R)

        h= [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        # print('h len:', len(h), 'h0', h[0].shape)
        return h
