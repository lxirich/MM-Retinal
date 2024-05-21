import os
from abc import abstractmethod
import json
import time
import torch
import pandas as pd
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        ## record all the val
        record_json = {}
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, test_gts, test_res = self._train_epoch(epoch)
            if test_gts is not None:
                # save outputs each epoch
                save_outputs = {'gts': test_gts, 'res': test_res}
                save_path = os.path.join(self.args.record_dir, 'Epoch', str(epoch)+'_token_results.json')
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                with open(os.path.join(save_path), 'w') as f:
                    json.dump(save_outputs, f)

                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)
                record_json[epoch] = log

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                        save_outputs = {'gts': test_gts, 'res': test_res}
                        with open(os.path.join(self.args.record_dir, 'best_word_results.json'), 'w') as f:
                            json.dump(save_outputs, f)
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()
        self._save_file(record_json)

    def _save_file(self, log):
        if not os.path.exists(self.args.record_dir):
            os.mkdir(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.json')
        with open(record_path, 'w') as f:
            json.dump(log, f)

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        val_df = pd.DataFrame([self.best_recorder['val']])
        test_df = pd.DataFrame([self.best_recorder['test']])

        # Concat DataFrames
        record_table = pd.concat([record_table, val_df, test_df], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')

        if cuda_visible_devices != '':
            n_gpu_use = len(cuda_visible_devices.split(','))
        else:
            n_gpu_use = torch.cuda.device_count()

        if n_gpu_use == 0:
            print("Warning: There's no GPU available on this machine, training will be performed on CPU.")
        else:
            print(f"Using {n_gpu_use} GPU(s), GPUs numbers is {cuda_visible_devices.split(',')}.")

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        ## check the training
        self.writer = SummaryWriter()

    def _train_epoch(self, epoch):

        train_loss = 0
        print_loss = 0

        self.model.train()
        loop = tqdm(enumerate(self.train_dataloader), leave=True, total=len(self.train_dataloader))
        for batch_idx, (images_id, images, reports_ids, reports_masks) in loop:
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
            output = self.model(images=images, targets=reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item() / self.args.batch_size
            self.writer.add_scalar("data/Loss", loss.item(), batch_idx+len(self.train_dataloader)*(epoch-1))
            # To activate the tensorboard: tensorboard --logdir=runs --bind_all
            print_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            loop.set_description(f'Train Epoch [{epoch}/{self.args.epochs}]')
            # Show the loss in current batch
            loop.set_postfix(loss=loss.item()/self.args.batch_size)
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print("Finish Epoch {} Training, Start Eval...".format(epoch))

        # Determine the round at which validation starts
        val_epoch = 3 if self.args.testing else 9
        # Determine whether to perform validation and testing
        # if epoch < val_epoch:
        #     return log, None, None
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            loop = tqdm(enumerate(self.val_dataloader), leave=True, total=len(self.val_dataloader))
            for batch_idx, (images_id, images, reports_ids, reports_masks) in loop:
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images=images, mode='sample')
                if self.args.n_gpu > 1:
                    tokenizer = self.model.module.tokenizer
                else:
                    tokenizer = self.model.tokenizer
                reports = tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                loop.set_description(f'Valid Epoch [{epoch}/{self.args.epochs}]')

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            
        self.writer.add_scalar("data/b1/val", val_met['BLEU_1'], epoch)
        self.writer.add_scalar("data/b2/val", val_met['BLEU_2'], epoch)
        self.writer.add_scalar("data/b3/val", val_met['BLEU_3'], epoch)
        self.writer.add_scalar("data/b4/val", val_met['BLEU_4'], epoch)
        self.writer.add_scalar("data/met/val", val_met['METEOR'], epoch)
        self.writer.add_scalar("data/rou/val", val_met['ROUGE_L'], epoch)
        self.writer.add_scalar("data/cid/val", val_met['CIDER'], epoch)

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            loop = tqdm(enumerate(self.test_dataloader))
            for batch_idx, (images_id, images, reports_ids, reports_masks) in loop:
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images=images, mode='sample')
                if self.args.n_gpu > 1:
                    tokenizer = self.model.module.tokenizer
                else:
                    tokenizer = self.model.tokenizer
                reports = tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                loop.set_description(f'Test Epoch [{epoch}/{self.args.epochs}]')

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        self.writer.add_scalar("data/b1/test", test_met['BLEU_1'], epoch)
        self.writer.add_scalar("data/b2/test", test_met['BLEU_2'], epoch)
        self.writer.add_scalar("data/b3/test", test_met['BLEU_3'], epoch)
        self.writer.add_scalar("data/b4/test", test_met['BLEU_4'], epoch)
        self.writer.add_scalar("data/met/test", test_met['METEOR'], epoch)
        self.writer.add_scalar("data/rou/test", test_met['ROUGE_L'], epoch)
        self.writer.add_scalar("data/cid/test", test_met['CIDER'], epoch)

        self.lr_scheduler.step()
        self.writer.close()

        return log, test_gts, test_res
