import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # General Args
    parser.add_argument('--rs_type', type=str, default='General',
                        choices=['Seq', 'LLM', 'General'],
                        help='Seq, LLM, General')
    parser.add_argument('--model_name', type=str, default='SASRec',
                        help='model name.')
    parser.add_argument('--dataset', nargs='?', default='yc',
                        help='yc, ks, rr')

    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--test_only', action="store_true",
                        help='Whether to test only.')
    parser.add_argument('--clear_checkpoints', action="store_true",
                        help='Whether clear the earlier checkpoints.')
    parser.add_argument('--saveID', type=str, default='Saved',
                        help='Specify model save path. Description of the experiment')
    
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--verbose', type=float, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping point.')

    parser.add_argument("--mix", action="store_true",
                        help="whether to use mixed dataset")

    # Model Args
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    # parser.add_argument('--batch_size', type=int, default=128,
                        # help='Batch size.')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='Learning rate.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay for optimizer.')
    parser.add_argument("--no_wandb", action="store_true",
                        help="whether to use wandb")

    args, _ = parser.parse_known_args()

    if(args.rs_type == 'General'):
        parser.add_argument("--candidate", action="store_true",
                            help="whether using the candidate set")
        parser.add_argument('--Ks', type = int, default= 20,
                            help='Evaluate on Ks optimal items.')
        parser.add_argument('--neg_sample',type=int,default=1)
        parser.add_argument('--infonce', type=int, default=0,
                    help='whether to use infonce loss or not')
        parser.add_argument("--train_norm", action="store_true",
                            help="train_norm")
        parser.add_argument("--pred_norm", action="store_true",
                            help="pred_norm")
        parser.add_argument('--data_path', nargs='?', default='./data/General/',
                            help='Input data path.')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='number of workers in data loader')
        parser.add_argument('--regs', type=float, default=1e-5,
                            help='Regularization.')
        parser.add_argument('--max2keep', type=int, default=1,
                            help='max checkpoints to keep')
        args, _ = parser.parse_known_args()
        
        
        # InfoNCE
        if(args.model_name == 'InfoNCE'):
            parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature parameter')

        #AdvInfoNCE
        if(args.model_name == 'AdvInfoNCE'):
            parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature parameter')
            parser.add_argument('--eta_epochs', type=int, default=7,
                                help='epochs for eta, control the disturbance of adv training')
            parser.add_argument('--adv_lr', type=float, default=5e-5,
                                help='Learning rate for adversarial training.')
            parser.add_argument('--model_version', type=str, default='embed',
                                help='model type, mlp or embed')
            
            parser.add_argument('--adv_interval',type=int,default=5,
                                help='the interval of adversarial training')
            parser.add_argument('--warm_up_epochs', type=int, default=0,
                                help='warm up epochs, in this stage, adv training is not used')
            parser.add_argument('--k_neg', type=float, default=64,
                                help='k_neg for negative sampling')
            parser.add_argument('--adv_epochs',type=int,default=1,
                                help='the epoch of adversarial training')
            parser.add_argument('--w_embed_size',type=int,default=64,
                                help='dimension of weight embedding')
        
        # MultVAE
        if(args.model_name == 'MultVAE'):
            parser.add_argument('--total_anneal_steps', type=int, default=200000,
                            help='total anneal steps')
            parser.add_argument('--anneal_cap', type=float, default=0.2,
                            help='anneal cap')
            parser.add_argument('--p_dim0', type=int, default=200,
                            help='p_dim0')
            parser.add_argument('--p_dim1', type=int, default=600,
                            help='p_dim1')

        # AlphaRec
        if args.model_name == "AlphaFree" or args.model_name == "AlphaFree_col":
            parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature parameter')
            parser.add_argument('--lm_model', type=str, default='v3',
                        choices=['bert', 'roberta', 'llama2_7b', 'llama3_7b', 'mistral_7b', 'v2', 'v3', 'SFR', 'v3_shuffle', 'xavior', 'mist', 'llama'],
                        help='The base language model')
            parser.add_argument('--align', type=float, default=0,
                        help='The mapping method')
            parser.add_argument('--topk_knn', type=int, default=0,
                        help='The mapping method')
            parser.add_argument('--align_temperature', type=float, default=0.07,
                                help='align cl loss temperature')

    args_full, _ = parser.parse_known_args()
    special_args = list(set(vars(args_full).keys()) - set(vars(args).keys()))
    special_args.sort()
    return args_full, special_args
