import os
import torch
import wandb
import time

from data_loader.MSC_data_loader import MSC_Dataset
from torch.utils.data import DataLoader
from model.architecture import MyArch
from tqdm import tqdm


class MyTrainer:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path

    def train_with_hyper_param(self, args):
        # save dir create
        checkpoint_directory = f"../checkpoint/{args.data_name}/"
        try:
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
        except OSError:
            print("Error: Failed to create the directory.")
            exit()
                
        # model load
        model = MyArch(args)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay_rate)
        
        if not args.debug:  # code for debug mode
            wandb.init(project="Template Project", resume=True)
            wandb.run.name = "Template run" + str(time.time())
            if wandb.run.resumed:
                checkpoint = torch.load(wandb.restore(f"{checkpoint_directory}/{str(args.resume)}_epochs.tar"))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = args.resume
                
        model.to(self.device)
        
        # if args.LLM_freeze:
        #     for parameters in model.generator_model.parameters():
        #         parameters.requires_grad = False

        # data load
        if args.data_name == 'MSC':
            train_dataset = MSC_Dataset(self.data_path, mode='train', device=self.device, args=args)
            valid_dataset = MSC_Dataset(self.data_path, mode='valid', device=self.device, args=args)
            
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

        train_batch_len = len(train_dataloader)
        valid_batch_len = len(valid_dataloader)

        pbar = tqdm(range(1, args.epochs+1 - args.resume), position=0, leave=False, desc='epoch')
        for epoch in pbar:
            total_train_loss = 0
            total_valid_loss = 0
            
            # training
            model.train()
            prog_bar = tqdm(train_dataloader, position=1,leave=False, desc='batch')
            for i, (inputs, labels) in enumerate(prog_bar):
                input = [ i.to(self.device) for i in inputs ]
                label = [ l.to(self.device) for l in labels ]
                
                optimizer.zero_grad()
                
                loss = model(input, label)
                
                prog_bar.set_postfix({'loss': loss.item()})
                
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                if not args.debug:  # code for debug mode
                    if i % (100//args.batch_size) == 0:
                        wandb.log({'train_loss': loss.item()})
            
            # validation
            with torch.no_grad():
                model.eval()
                metric_log = (epoch) % args.metric_at_every == 0
                for inputs, labels in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    inputs = [ i.to(self.device) for i in inputs ]
                    labels = [ l.to(self.device) for l in labels ]
                    
                    loss = model(inputs, labels, metric_log=metric_log, epoch=epoch)
                    total_valid_loss += loss.item()

            # log
            output_loss__dict = {'train_loss (epoch)': total_train_loss/train_batch_len,
                                 'valid_loss (epoch)': total_valid_loss/valid_batch_len
                                 }
            
            if metric_log:
                eval_result = self.get_metrics()

                output_metric_dict = {'valid_Bleu-3 (epoch)': eval_result['Bleu_3'],
                                      'valid_CIDEr (epoch)': eval_result['CIDEr'],
                                      }
            
            if not args.debug:  # code for debug mode
                wandb.log(output_loss__dict)
                pbar.write("output_loss__dict: ", output_loss__dict)
                if metric_log:
                    wandb.log(output_metric_dict)
                    pbar.write("output_metric_dict: ", output_metric_dict)
            
            # save checkpoint
            if (epoch) % args.save_at_every == 0:
                CHECKPOINT_PATH = f"{checkpoint_directory}/{str(epoch)}_epochs.tar"
                torch.save({ # Save our checkpoint loc
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, CHECKPOINT_PATH)
                wandb.save(CHECKPOINT_PATH)
                pbar.write('Checkpoint model has saved at Epoch: {:02} '.format(epoch))
                

            scheduler.step()  # per epochs
        pbar.close()

        return model

    def get_metrics(self):
        eval_result = {}
        eval_result['Bleu_3'] = 0
        eval_result['CIDEr'] = 0
        
        return eval_result
