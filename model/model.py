import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netP = self.set_device(networks.define_P(opt))
        self.netG = self.set_device(networks.define_G(opt))
        self.netguide_3D_1 = self.set_device(networks.define_guide(opt,5,1))
        self.netguide_3D_2 = self.set_device(networks.define_guide(opt,5,2))
        self.netguide_3D_3 = self.set_device(networks.define_guide(opt,5,3))

        self.netguide_spectrum_1 = self.set_device(networks.define_guide(opt,1,1))
        self.netguide_spectrum_2 = self.set_device(networks.define_guide(opt,1,2))
        self.netguide_spectrum_3 = self.set_device(networks.define_guide(opt,1,3))
        self.schedule_phase = None
        self.lr = opt['train']["optimizer"]["lr"]
        # set loss and load resume state
        self.loss_func = nn.L1Loss(reduction='sum').to(self.device)
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')

        optim_params = list(self.netG.parameters())
        optim_params_P = list(self.netP.parameters())
        optim_params_guide_3D_1 = list(self.netguide_3D_1.parameters())
        optim_params_guide_3D_2 = list(self.netguide_3D_2.parameters())
        optim_params_guide_3D_3 = list(self.netguide_3D_3.parameters())

        optim_params_guide_spectrum_1 = list(self.netguide_spectrum_1.parameters())
        optim_params_guide_spectrum_2 = list(self.netguide_spectrum_2.parameters())
        optim_params_guide_spectrum_3 = list(self.netguide_spectrum_3.parameters())
            # optim_params_guide_spectrum_4 = list(self.netguide_spectrum_4.parameters())
        self.optG = torch.optim.Adam(
            optim_params, lr=opt['train']["optimizer"]["lr"] , weight_decay=0.0001)
        self.optP = torch.optim.Adam(
            optim_params_P, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
        self.optguide_3D_1 = torch.optim.Adam(
            optim_params_guide_3D_1, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
        self.optguide_3D_2 = torch.optim.Adam(
            optim_params_guide_3D_2, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
        self.optguide_3D_3 = torch.optim.Adam(
            optim_params_guide_3D_3, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)

        self.optguide_spectrum_1 = torch.optim.Adam(
            optim_params_guide_spectrum_1, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
        self.optguide_spectrum_2 = torch.optim.Adam(
            optim_params_guide_spectrum_2, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
        self.optguide_spectrum_3 = torch.optim.Adam(
            optim_params_guide_spectrum_3, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)

        self.log_dict = OrderedDict()
        self.load_network()

    def feed_data(self, data):
        self.data = self.set_device(data)
    def guide_predict(self):
        _,loss1 = self.netguide_3D_1(self.data['L3D'],self.data['H3D'],t=None)
        _,loss2 = self.netguide_3D_2(self.data['L3D'],self.data['H3D'],t=None)
        _,loss3 = self.netguide_3D_3(self.data['L3D'],self.data['H3D'],t=None)

        _,loss9 = self.netguide_spectrum_1(self.data['LP'],self.data['HP'],t=None)
        _,loss10 = self.netguide_spectrum_2(self.data['LP'],self.data['HP'],t=None)
        _,loss11 = self.netguide_spectrum_3(self.data['LP'],self.data['HP'],t=None)
        # _,loss12 = self.netguide_spectrum_4(self.data['LP'],self.data['HP'],t=None)
        ax_feature=[self.netguide_3D_1.get_feature(),self.netguide_3D_2.get_feature(),self.netguide_3D_3.get_feature()]

        fr_feature=[self.netguide_spectrum_1.get_feature(),self.netguide_spectrum_2.get_feature(),self.netguide_spectrum_3.get_feature()]
        loss = loss1+loss2+loss3+loss9+loss10+loss11
        return ax_feature,fr_feature,loss

    def optimize_parameters(self):
        self.optG.zero_grad()
        self.optP.zero_grad()
        self.optguide_3D_1.zero_grad()
        self.optguide_3D_2.zero_grad()
        self.optguide_3D_3.zero_grad()

        self.optguide_spectrum_1.zero_grad()
        self.optguide_spectrum_2.zero_grad()
        self.optguide_spectrum_3.zero_grad()
        ax_feature, fr_feature, loss_guide = self.guide_predict()
        self.initial_predict(ax_feature, fr_feature)
        # calculate residual as x_start
        self.data['IP'] = self.IP
        self.data['RS'] = self.data['HR'] - self.IP
        l_pix, l_cdcd = self.netG(self.data,ax_feature, fr_feature)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = (l_pix.sum())/int(b*c*h*w)+loss_guide
        l_pix.backward()
        # update all networks
        self.optG.step()
        self.optP.step()
        self.optguide_3D_1.step()
        self.optguide_3D_2.step()
        self.optguide_3D_3.step()

        self.optguide_spectrum_1.step()
        self.optguide_spectrum_2.step()
        self.optguide_spectrum_3.step()
        # set log
        self.log_dict['l_total'] = l_pix.item()
        self.log_dict['guide_loss'] = loss_guide.item()
    def initial_predict(self,ax_feature, fr_feature):
        self.IP = self.netP(self.data['SR'],time = None, ax_feature=ax_feature, fr_feature=fr_feature)

    def test(self, continous=False):
        self.netG.eval()
        self.netP.eval()
        self.netguide_3D_1.eval()
        self.netguide_3D_2.eval()
        self.netguide_3D_3.eval()
        self.netguide_spectrum_1.eval()
        self.netguide_spectrum_2.eval()
        self.netguide_spectrum_3.eval()
        ax_feature, fr_feature, _ = self.guide_predict()
        with torch.no_grad():
            self.IP = self.netP(self.data['SR'], time=None, ax_feature=ax_feature, fr_feature=fr_feature)
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], ax_feature, fr_feature, continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], ax_feature, fr_feature, continous)
        self.netG.train()
        self.netP.train()
        self.netguide_3D_1.train()
        self.netguide_3D_2.train()
        self.netguide_3D_3.train()
        self.netguide_spectrum_1.train()
        self.netguide_spectrum_2.train()
        self.netguide_spectrum_3.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)


    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['IP'] = self.IP.detach().float().cpu()
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_state_dict(self, net, iter_step, epoch, name):
        if isinstance(net, nn.DataParallel):
            net = net.module
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        gen_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_{name}_gen.pth')
        torch.save(state_dict, gen_path)
        return gen_path

    def save_optimizer_state(self, opt_net, iter_step, epoch, name):
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': opt_net.state_dict()}
        opt_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_{name}_opt.pth')
        torch.save(opt_state, opt_path)

    def save_network(self, epoch, iter_step):
        networks = [
            (self.netP, self.optP, "PreNet"),
            (self.netG, self.optG, "DenoiseNet"),
            (self.netguide_3D_1, self.optguide_3D_1, "guide_3D_1"),
            (self.netguide_3D_2, self.optguide_3D_2, "guide_3D_2"),
            (self.netguide_3D_3, self.optguide_3D_3, "guide_3D_3"),
            (self.netguide_spectrum_1, self.optguide_spectrum_1, "guide_spectrum_1"),
            (self.netguide_spectrum_2, self.optguide_spectrum_2, "guide_spectrum_2"),
            (self.netguide_spectrum_3, self.optguide_spectrum_3, "guide_spectrum_3"),
        ]

        for net, opt_net, name in networks:
            gen_path = self.save_state_dict(net, iter_step, epoch, name)
            self.save_optimizer_state(opt_net, iter_step, epoch, name)

        logger.info(f'Saved model in [{gen_path}] ...')

    def load_network_state(self, network, load_path, model_name):
        gen_path = f'{load_path}_{model_name}_gen.pth'
        logger.info(f'Loading pretrained model for {model_name} [{gen_path}] ...')
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
        return network

    def load_optimizer_state(self, opt_net, load_path, model_name):
        opt_path = f'{load_path}_{model_name}_opt.pth'
        opt = torch.load(opt_path)
        opt_net.load_state_dict(opt['optimizer'])
        self.begin_step = opt['iter']
        self.begin_epoch = opt['epoch']

    def load_network(self):
        if self.opt['path']['resume_state'] is not None:
            load_path = self.opt['path']['resume_state']
            networks = [
                (self.netP, self.optP, "PreNet"),
                (self.netG, self.optG, "DenoiseNet"),
                (self.netguide_3D_1, self.optguide_3D_1, "guide_3D_1"),
                (self.netguide_3D_2, self.optguide_3D_2, "guide_3D_2"),
                (self.netguide_3D_3, self.optguide_3D_3, "guide_3D_3"),
                (self.netguide_spectrum_1, self.optguide_spectrum_1, "guide_spectrum_1"),
                (self.netguide_spectrum_2, self.optguide_spectrum_2, "guide_spectrum_2"),
                (self.netguide_spectrum_3, self.optguide_spectrum_3, "guide_spectrum_3"),
            ]

            for net, opt_net, name in networks:
                net = self.load_network_state(net, load_path, name)
                if self.opt['phase'] == 'train':
                    self.load_optimizer_state(opt_net, load_path, name)
