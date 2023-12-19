import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from src.callbacks.ema import EMACallback
from pytorch_lightning.plugins import DDPPlugin
from src.utils.torch_dist import all_gather_object, synchronize

from .nuscenes.base_exp import VAMPIRELightningModel

def run_cli(model_class=VAMPIRELightningModel,
            exp_name='base_exp',
            use_ema=False,
            extra_trainer_config_args={}):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-v',
                               '--validate',
                               dest='validate',
                               action='store_true',
                               help='validate model on validation set')
    parent_parser.add_argument('-t',
                               '--test',
                               dest='test',
                               action='store_true',
                               help='test model on validation set (output submit file)')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set (output submit file)')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--debug',
                               action='store_true',
                               default=False,
                               help='debug')
    parent_parser.add_argument('--vis',
                               action='store_true',
                               default=False,
                               help='test model on visualization set, output vis file')
    parent_parser.add_argument('--trainval',
                               action='store_true',
                               default=False,
                               help='training using trainval set')
    parent_parser.add_argument('--ckpt_path', type=str)
    debug_now = parent_parser.parse_args().debug
    trainval = parent_parser.parse_args().trainval
    vis = parent_parser.parse_args().vis
    parser = VAMPIRELightningModel.add_model_specific_args(parent_parser)
    if debug_now:
        parser.set_defaults(profiler='simple',
                            deterministic=False,
                            max_epochs=extra_trainer_config_args.get('epochs', 24),
                            accelerator='cpu',
                            num_sanity_val_steps=0,
                            gradient_clip_val=35,
                            # limit_val_batches=1.,
                            enable_checkpointing=True,
                            # precision=16,
                            # sync_batchnorm=True,
                            default_root_dir=os.path.join('./outputs/', exp_name))
    else:
        if trainval:
            parser.set_defaults(profiler='simple',
                                deterministic=False,
                                max_epochs=extra_trainer_config_args.get('epochs', 24),
                                accelerator='ddp',
                                num_sanity_val_steps=0,
                                gradient_clip_val=35,
                                limit_val_batches=0,
                                enable_checkpointing=True,
                                precision=16,
                                sync_batchnorm=True,
                                default_root_dir=os.path.join('./outputs/', exp_name))
        else:
            parser.set_defaults(profiler='simple',
                                deterministic=False,
                                max_epochs=extra_trainer_config_args.get('epochs', 24),
                                accelerator='ddp',
                                num_sanity_val_steps=0,
                                gradient_clip_val=35,
                                # limit_val_batches=1.,
                                check_val_every_n_epoch=4,
                                enable_checkpointing=True,
                                precision=16,
                                sync_batchnorm=True,
                                default_root_dir=os.path.join('./outputs/', exp_name))
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))

    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback])
    else:
        trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=False))
    if args.validate:
        model = model.load_from_checkpoint(args.ckpt_path, strict=False)
        trainer.validate(model)
    elif args.test:
        model = model.load_from_checkpoint(args.ckpt_path, strict=False)
        trainer.test(model)
    elif args.predict:
        model = model.load_from_checkpoint(args.ckpt_path, strict=False)
        predict_step_outputs = trainer.predict(model)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
    else:
        if args.ckpt_path is not None:
            model = model.load_from_checkpoint(args.ckpt_path)
            trainer.fit(model)
            # trainer.fit(model, ckpt_path=args.ckpt_path)
        else:
            trainer.fit(model)
