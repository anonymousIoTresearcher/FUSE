import  torch, os
import  numpy as np
from    FUSENShot import FUSENShot
import  argparse
import pickle
from    meta_regression import Meta

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [16, 5, 3, 3, 1, 1]),
        ('relu', [True]),
        # ('bn', [32]),
        ('conv2d', [32, 16, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('flatten', []),
        ('linear', [512, 6272]),
        ('bn', [512]),
        ('linear', [57, 512])
    ]

    device = torch.device('cuda:1')
    maml = Meta(args, config).to(device)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #maml = Meta(args, config)
    #if torch.cuda.device_count() > 1:
    #    maml = torch.nn.DataParallel(maml,device_ids=[0,1])
     
    #maml.to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = FUSENShot('FUSE',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)
    best_mae = 99999
    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        maes = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 50 == 0:
            print('step:', step, '\ttraining mae:', maes)

        if step % 500 == 0:
            maes = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_mae = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    # print("test_mae is: ", test_mae)
                    maes.append( test_mae )

            # [b, update_step+1]
            maes = np.array(maes).mean(axis=0).astype(np.float16)
            print('Test mae:', maes)            
            if maes[-1] < best_mae:
                best_mae = maes[-1]
                model_params = [item.cpu().detach().numpy() for item in maml.net.vars]
                model_params_file = open('FUSE/model/model_param_best_%sway%sshot.pkl' % (args.n_way, args.k_spt), 'wb')
                pickle.dump(model_params, model_params_file)
                model_params_file.close()
            
    print('The best mae on validation set is {}'.format(best_mae))



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=200)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=200)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=14)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=20)

    args = argparser.parse_args()

    main(args)
