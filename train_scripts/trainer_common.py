#!/usr/bin/env python3

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
from utilities.utilities import calculate_psnr, calculate_ssim,tensor2img

class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        dlModulename    = config["dataloader"]
        package         = __import__("data_tools.dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'GetLoader')
        #============build train dataloader==============#
        train_dataset   = config["dataset_paths"][config["dataset_name"]]
        print("Prepare the train dataloader...")
        self.train_loader     = dataloaderClass(train_dataset,
                                        config["batch_size"],
                                        config["random_seed"],
                                        **config["dataset_params"]
                                    )
 
        #========build evaluation dataloader=============#
        package         = __import__("data_tools.eval_dataloader", fromlist=True)
        dataloaderClass = getattr(package, 'EvalDataset')
        eval_dataset = config["test_dataset_paths"][config["dataset_name"]]
        print("Prepare the evaluation dataloader...")
        self.eval_loader      = dataloaderClass(
                                        config["dataset_name"],
                                        eval_dataset,
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"],
                                        subffix = config["dataset_params"]["subffix"]
                                    )
        self.eval_iter  = len(self.eval_loader)//config["eval_batch_size"]
        if len(self.eval_loader)%config["eval_batch_size"]>0:
            self.eval_iter+=1


    # TODO modify this function to build your models
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        # from components.RepSR_plain import RepSRPlain_pixel
        script_name     = "components."+self.config["module_script_name"]
        class_name      = self.config["class_name"]
        package         = __import__(script_name, fromlist=True)
        network_class   = getattr(package, class_name)

        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")

        # TODO replace below lines to define the model framework
        self.network = network_class(3,
                                    3,
                                    self.config["feature_num"],
                                    **self.config["module_params"]
                                    )
        self.reporter.writeModel(self.network.__str__())
        
        # print(self.network)


        # if in finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["project_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["ckpt"],
                                        self.config["checkpoint_names"]["generator_name"]))
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            model_spec = checkpoint['model_state_dict']
            own_state = self.network.state_dict()
            for name, param in model_spec.items():
                if name in own_state:
                    if isinstance(param, nn.Parameter):
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                    except Exception:
                        if name.find('tail') == -1:
                            raise RuntimeError('While copying the parameter named {}, '
                                            'whose dimensions in the model are {} and '
                                            'whose dimensions in the checkpoint are {}.'
                                            .format(name, own_state[name].size(), param.size()))
            print('loaded trained backbone model epoch {}...!'.format(self.config["ckpt"]))

        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()


    # TODO modify this function to evaluate your model
    def __evaluation__(self, eval_loader, eval_iter, epoch, step = 0):
        # Evaluate the checkpoint
        self.network.eval()
        total_psnr = 0
        total_ssim = 0
        total_num  = 0
        dataset_name = self.config["dataset_name"]
        patch_test = True
        with torch.no_grad():
            for _ in tqdm(range(eval_iter)):
                hr, lr = eval_loader()
                if self.config["cuda"] >=0:
                    hr = hr.cuda()
                    lr = lr.cuda()
                if patch_test:
                    tile = 64
                    tile_overlap = 24
                    scale = self.config["module_params"]["upsampling"]
                    b, c, h, w = lr.size()
                    tile = min(tile, h, w)

                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                    E = torch.zeros(b, c, h*scale, w*scale).type_as(lr)
                    W = torch.zeros_like(E)

                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = lr[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                            out_patch = self.network(in_patch)
                            if isinstance(out_patch, list):
                                out_patch = out_patch[-1]
                            out_patch_mask = torch.ones_like(out_patch)

                            E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                            W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
                    res = E.div_(W)
                else:
                    res = self.network(lr)
                # res     = self.network(lr)
                res     = tensor2img(res.cpu())
                hr      = tensor2img(hr.cpu())
                psnr    = calculate_psnr(res[0],hr[0])
                ssim    = calculate_ssim(res[0],hr[0])
                total_psnr+= psnr
                total_ssim+= ssim
                total_num+=1
        final_psnr = total_psnr/total_num
        final_ssim = total_ssim/total_num
        
        if (final_psnr>self.best_psnr["psnr"]):
            self.best_psnr["psnr"] = final_psnr
            self.best_psnr["epoch"] = epoch
            print("[{}], Best PSNR: {:.4f} @ epoch {}".format(self.config["version"],
                                                    self.best_psnr["psnr"], self.best_psnr["epoch"]))
            self.reporter.writeTrainLog(epoch,step, "Dataset: {}, Best PSNR: {:.4f}, SSIM: {:.4f}".format(dataset_name, final_psnr, final_ssim))

        print("[{}], Epoch [{}], Dataset: {}, PSNR: {:.4f}, SSIM: {:.4f}".format(self.config["version"], epoch, dataset_name, final_psnr, final_ssim))
        self.reporter.writeTrainLog(epoch,step, "Dataset: {}, PSNR: {:.4f}, SSIM: {:.4f}".format(dataset_name, final_psnr, final_ssim))
        

    # TODO modify this function to configurate the optimizer of your pipeline
    def __setup_optimizers__(self):
        train_opt = self.config['optim_config'] 
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.reporter.writeInfo(f'Params {k} will not be optimized.')

        optim_type = self.config['optim_type']
        if optim_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(optim_params,**train_opt)
        elif optim_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(optim_params,**train_opt)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        # self.optimizers.append(self.optimizer_g)
        

    def train(self):
        
        # general configurations 
        ckpt_dir    = self.config["project_checkpoints"]
        log_frep    = self.config["log_step"]
        model_freq  = self.config["model_save_epoch"]
        total_epoch = self.config["total_epoch"]
        l1_W        = self.config["l1_weight"]
        # lrDecayStep = self.config["lrDecayStep"]
        # TODO [more configurations here]
        self.best_psnr = {
            "epoch":-1,
            "psnr":-1
        }

        #===============build framework================#
        self.__init_framework__()
        # import pdb; pdb.set_trace()

        from thop import profile
        from thop import clever_format
        train_patch_size = self.config["dataset_params"]["lr_patch_size"]
        test_img    = torch.rand((1,3,train_patch_size,train_patch_size), device = 'cuda')

        macs, params = profile(self.network, inputs=(test_img,))
        macs, params = clever_format([macs, params], "%.3f")
        print("Model FLOPs: ",macs)
        print("Model Params:",params)
        self.reporter.writeInfo("Model FLOPs: "+macs)
        self.reporter.writeInfo("Model Params: "+params)

        # # set the start point for training loop
        # if self.config["phase"] == "finetune":
        #     start = self.config["checkpoint_epoch"] - 1
        # else:
        #     start = 0
        start = 0
        

        #===============build optimizer================#
        print("build the optimizer...")
        # Optimizer
        # TODO replace below lines to build your optimizer
        self.__setup_optimizers__()
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["project_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["ckpt"],
                                        self.config["checkpoint_names"]["generator_name"]))
            checkpoint = torch.load(model_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']

        #===============build losses===================#
        # TODO replace below lines to build your losses
        # l1 = nn.L1Loss() # [replace this]
        l1 = nn.MSELoss()

        
        # Caculate the epoch number
        step_epoch  = len(self.train_loader)
        print("Total step = %d in each epoch"%step_epoch)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print('Start   ===========================  training...')
        start_time = time.time()
        # import pdb; pdb.set_trace()
        for epoch in range(start, total_epoch):
            for step in range(step_epoch):
                # Set the networks to train mode

                self.network.train()
                # TODO [add more code here]
                # clear cumulative gradient
                self.optimizer.zero_grad()

                # TODO read the training data
                
                hr, lr  = self.train_loader.next()
                
                
                generated_hr = self.network(lr)

                loss_l1  = l1(generated_hr, hr)
                # loss_per = ploss(generated_hr, hr)
                loss_curr = loss_l1 #+ lambda_p*loss_per

                loss_curr.backward()

                self.optimizer.step()

                
                loss_cur_scalar = loss_curr.item()
                loss_l1_scalar = loss_l1.item()
                
                # Print out log info
                if (step + 1) % log_frep == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    # cumulative steps
                    cum_step = (step_epoch * epoch + step + 1)
                    
                    self.reporter.writeTrainLog(epoch+1,step+1,
                                "loss: {:.4f}, l1: {:.4f}".format(loss_cur_scalar, loss_l1_scalar))

                    
            
            #===============adjust learning rate============#
            if (epoch + 1) in self.config["lr_decay_step"] and self.config["lr_decay_enable"]:
                print("Learning rate decay")
                for p in self.optimizer.param_groups:
                    p['lr'] *= self.config["lr_decay"]
                    print("Current learning rate is %f"%p['lr'])

            #===============save checkpoints================#
            if (epoch+1) % model_freq==0:
                print("Save epoch %d model checkpoint!"%(epoch+1))
                torch.save(
                    {
                        "epoch" : epoch+1,
                        "model_state_dict" : self.network.state_dict(),
                        "optimizer_state_dict" : self.optimizer.state_dict(),
                    }, 
                    os.path.join(ckpt_dir, 'epoch{}_{}.pth'.format(epoch + 1, self.config["checkpoint_names"]["generator_name"]))
                )
                

                self.__evaluation__(self.eval_loader, self.eval_iter, epoch+1)
