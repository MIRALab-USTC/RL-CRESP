import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env', type=str, default='dmc.hopper.hop')
    # parser.add_argument('--action_repeat', '-ar', default=2, type=int)
    # algorithm
    parser.add_argument('--base', type=str, default='sac', choices=['sac', 'td3'])
    parser.add_argument('--agent', type=str, default='drq', choices=['drq', 'curl'])
    parser.add_argument('--auxiliary', type=str, default=None,
                        choices=['cresp'])
    # aug
    parser.add_argument('--data_aug', default='shift', type=str)
    # training settings
    parser.add_argument('--num_sources', default=2, type=int)
    parser.add_argument('--dynamic', default=True, action='store_true')
    parser.add_argument('--background', '-bg', default=False, action='store_true')
    parser.add_argument('--camera', '-ca', default=False, action='store_true')
    parser.add_argument('--color', '-co', default=False, action='store_true')
    parser.add_argument('--test_background', '-tbg', default=False, action='store_true')
    parser.add_argument('--test_camera', '-tca', default=False, action='store_true')
    parser.add_argument('--test_color', '-tco', default=False, action='store_true')
    parser.add_argument('--batch_size', default=512, type=int)

    """Reset the initial hyperparameters"""
    parser.add_argument('--disenable_default', default=False, action='store_true')
    # training hypers
    parser.add_argument('--nstep_of_rsd', default=5, type=int)
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument('--extr_latent_dim', default=50, type=int)
    parser.add_argument('--extr_update_via_qfloss', default='True')
    parser.add_argument('--extr_update_freq_via_qfloss', default=1, type=int)
    parser.add_argument('--actor_update_mode', default='sum', type=str)
    # module
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--alpha_lr', default=5e-4, type=float)
    parser.add_argument('--extr_lr', default=5e-4, type=float)
    parser.add_argument('--targ_extr', default=False, action='store_true')
    parser.add_argument('--discount_of_rs', default=0.8, type=float)
    parser.add_argument('--num_sample', default=256, type=int)
    parser.add_argument('--num_ensemble', default=7, type=int)
    parser.add_argument('--opt_num', default=5, type=int)
    parser.add_argument('--opt_mode', default='min', type=str)
    parser.add_argument('--omega_opt_mode', default='min_mu',
                        choices=[None, 'min_mu', 'min_all'])
    parser.add_argument('--rs_fc', default=False, action='store_true')
    # save
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    # seed
    parser.add_argument('--seed_list', '-s', nargs='+', type=int,
                         default=[0, 1, 2, 3, 4])
    parser.add_argument('--seed', default=1, type=int)
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--cuda_id', type=int, default=0)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.extr_update_via_qfloss in ['True', 'False']
    args.extr_update_via_qfloss = True if args.extr_update_via_qfloss == 'True' else False
    return args
