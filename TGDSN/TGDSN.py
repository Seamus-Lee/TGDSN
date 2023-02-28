import sys
sys.path.append("..")
from utils import *
from mydataset import create_dataset_full
from models import *
from train_eval import *
import time
import itertools



def TGDSN(params, device, config, model, my_dataset, src_id, tgt_id, args):
    hyper = params[f'{src_id}_{tgt_id}']
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    denorm = 130
    batch_size = hyper['batch_size']
    constract_vision = 2

    manual_seed = random.randint(1, 10000)
    print(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)

    ''' Load data '''
    src_train_dl, src_test_dl = create_dataset_full(my_dataset[int(src_id[-1]) - 1], batch_size=hyper['batch_size'])
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[int(tgt_id[-1]) - 1], batch_size=hyper['batch_size'])


    if src_id == 'FD001' and tgt_id == 'FD003':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_16.pt',
            map_location=torch.device('cpu'))
        private_encoder_s = TGAT(hidden_dim=16).to(device)
        private_encoder_t = TGAT(hidden_dim=16).to(device)
        decoder_s = InnerProductDecoder().to(device)
        decoder_t = InnerProductDecoder().to(device)

        ''' shared encoder '''
        shared_encoder = TGAT(hidden_dim=16).to(device)
        shared_encoder.load_state_dict(checkpoint['state_dict'])
        s_model = shared_encoder.model
        s_att = shared_encoder.dense_weight

    elif src_id == 'FD001' and tgt_id == 'FD004':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_16.pt',
            map_location=torch.device('cpu'))
        private_encoder_s = TGAT(hidden_dim=16).to(device)
        private_encoder_t = TGAT(hidden_dim=16).to(device)
        decoder_s = InnerProductDecoder().to(device)
        decoder_t = InnerProductDecoder().to(device)

        ''' shared encoder '''
        shared_encoder = TGAT(hidden_dim=16).to(device)
        shared_encoder.load_state_dict(checkpoint['state_dict'])
        s_model = shared_encoder.model
        s_att = shared_encoder.dense_weight

    elif src_id == 'FD002' and tgt_id == 'FD003':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_16.pt',
            map_location=torch.device('cpu'))
        private_encoder_s = TGAT(hidden_dim=16).to(device)
        private_encoder_t = TGAT(hidden_dim=16).to(device)
        decoder_s = InnerProductDecoder().to(device)
        decoder_t = InnerProductDecoder().to(device)

        ''' shared encoder '''
        shared_encoder = TGAT(hidden_dim=16).to(device)
        shared_encoder.load_state_dict(checkpoint['state_dict'])
        s_model = shared_encoder.model
        s_att = shared_encoder.dense_weight

    elif src_id == 'FD004' and tgt_id == 'FD001':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_16.pt',
            map_location=torch.device('cpu'))
        private_encoder_s = TGAT(hidden_dim=16).to(device)
        private_encoder_t = TGAT(hidden_dim=16).to(device)
        decoder_s = InnerProductDecoder().to(device)
        decoder_t = InnerProductDecoder().to(device)

        ''' shared encoder '''
        shared_encoder = TGAT(hidden_dim=16).to(device)
        shared_encoder.load_state_dict(checkpoint['state_dict'])
        s_model = shared_encoder.model
        s_att = shared_encoder.dense_weight

    elif src_id == 'FD002' and tgt_id == 'FD004':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_24.pt',
            map_location=torch.device('cpu'))
        private_encoder_s = TGAT(hidden_dim=24).to(device)
        private_encoder_t = TGAT(hidden_dim=24).to(device)
        decoder_s = InnerProductDecoder().to(device)
        decoder_t = InnerProductDecoder().to(device)

        ''' shared encoder '''
        shared_encoder = TGAT(hidden_dim=24).to(device)
        shared_encoder.load_state_dict(checkpoint['state_dict'])
        s_model = shared_encoder.model
        s_att = shared_encoder.dense_weight
    else:
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_32.pt',
            map_location=torch.device('cpu'))


        private_encoder_s = TGAT(hidden_dim=32).to(device)
        private_encoder_t = TGAT(hidden_dim=32).to(device)
        decoder_s = InnerProductDecoder().to(device)
        decoder_t = InnerProductDecoder().to(device)

        ''' shared encoder '''
        shared_encoder = TGAT(hidden_dim=32).to(device)
        shared_encoder.load_state_dict(checkpoint['state_dict'])
        s_model = shared_encoder.model
        s_att = shared_encoder.dense_weight


    ''' Load adj labels for reconstruction '''
    src_adj_path = "D:/TGDSN/data/{}_adj.pt".format(src_id)
    src_adj = torch.load(src_adj_path)

    tgt_adj_path = "D:/TGDSN/data/{}_adj.pt".format(tgt_id)
    tgt_adj = torch.load(tgt_adj_path)

    def adj_cross(src_adj, tgt_adj):
        A = torch.zeros(14, 14)
        for i in range(14):
            for j in range(14):
                if (src_adj[i][j] == 1 and tgt_adj[i][j] == 1):
                    A[i][j] = 1
        return A

    adj_shared = adj_cross(src_adj, tgt_adj)


    models = [private_encoder_s, private_encoder_t, s_model, s_att,decoder_s,decoder_t]
    params = itertools.chain(*[model.parameters() for model in models])

    # criterion
    criterion = RMSELoss()
    loss_diff = DiffLoss()

    optimizer = torch.optim.Adam(params, lr=hyper['lr'], weight_decay=5e-4)
    if hyper['StepLR'] == True:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,hyper['gamma'])

    best_loss = 50
    best_epoch = 0
    best_score_epoch = 0
    best_score = 100000
    best_loss_score = 100000
    test_loss_res = []
    rul_loss_all = []
    mmd_loss_all = []
    recon_all = []



    for epoch in range(1, hyper['epochs'] + 1):
        batch_iterator = zip(loop_iterable(src_train_dl), loop_iterable(tgt_train_dl))
        total_recon = 0
        total_grl = 0
        total_rul = 0
        start_time = time.time()

        len_dataloader = min(len(src_train_dl), len(tgt_train_dl))

        for i in range(len_dataloader):

            (source_x, source_label), (target_x, _) = next(batch_iterator)



            source_x = source_x.type_as(torch.FloatTensor())
            target_x = target_x.type_as(torch.FloatTensor())

            for model_train in models:
                model_train.train()
            optimizer.zero_grad()

            source_x, target_x = source_x.to(device), target_x.to(device)

            _, source_features_p,ps_output = private_encoder_s(src_adj, source_x)
            _, target_features_p,pt_output = private_encoder_t(tgt_adj, target_x)

            source_pred, source_features_s,source_output = shared_encoder(src_adj, source_x)

            _, target_features_s,target_output = shared_encoder(tgt_adj, target_x)



            ''' compute encoder difference loss for S and T '''

            diff_loss_s = loss_diff(source_features_p.float(), source_features_s.float())
            diff_loss_t = loss_diff(target_features_p.float(), target_features_s.float())
            diff_loss_all = diff_loss_s + diff_loss_t

            ''' compute decoder reconstruction loss for S and T '''

            if constract_vision == 1:
                # decoder code 版本1
                recovered_adj_s_p = decoder_s(source_features_p.float(), 256)
                recovered_adj_s_s = decoder_s(source_features_s.float(), 256)

                recovered_adj_t_p = decoder_t(target_features_p.float(), 256)
                recovered_adj_t_s = decoder_t(target_features_s.float(), 256)

                recon_loss_s = recon_loss(batch_size, recovered_adj_s_p, recovered_adj_s_s, A_metrix[int(src_id[-1]) - 1])
                recon_loss_t = recon_loss(batch_size, recovered_adj_t_p, recovered_adj_t_s, A_metrix[int(tgt_id[-1]) - 1])

            # decoder code 版本1结束

            if constract_vision == 2:
                # decoder code 版本2

                source_features_cat = torch.cat((source_features_s, source_features_p), 2)
                target_features_cat = torch.cat((target_features_s, target_features_p), 2)

                source_features_cat_adj = decoder_s(source_features_cat, 256)
                target_features_cat_adj = decoder_t(target_features_cat, 256)

                recon_loss_s = recon_loss_2(batch_size, source_features_cat_adj, src_adj)
                recon_loss_t = recon_loss_2(batch_size, target_features_cat_adj, tgt_adj)

            recon_loss_all = recon_loss_s + recon_loss_t

            ''' compute node classification loss for S '''


            pred = source_pred * denorm
            pred = pred.to(torch.float32)
            labels = source_label * denorm
            labels = labels.to(torch.float32)


            rul_loss = criterion(pred.squeeze(), labels.cuda())
            score = scoring_func(pred.squeeze() - labels.cuda())



            ''' compute domain classifier loss for both S and T '''
            # discriminator_x_s = source_features_s.float()
            # discriminator_x_t = target_features_s.float()
            discriminator_x_s = source_output.float()
            discriminator_x_t = target_output.float()

            loss_grl = mmd_rbf(discriminator_x_s, discriminator_x_t.float())


            ''' compute overall loss '''

            loss = hyper['lambda_rul']*rul_loss + hyper['lambda_d'] * loss_grl + hyper['lambda_r'] * recon_loss_all + hyper['lambda_f'] * diff_loss_all

            total_rul += rul_loss.item()
            total_grl += loss_grl.item()
            total_recon += recon_loss_all.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if hyper['StepLR'] == True:
            scheduler.step()
            print(f'scheduler_dis.get_last_lr:{scheduler.get_last_lr()[0]}')

        mean_grl = total_grl / (len_dataloader)
        mean_rul = total_rul / (len_dataloader)
        mean_recon = total_recon / (len_dataloader)



        rul_loss_all.append(mean_rul)
        mmd_loss_all.append(mean_grl)
        recon_all.append(mean_recon)


        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')

        print('rul_loss:{}'.format(mean_rul))
        print("loss_grl:{}".format(mean_grl))
        print("recon_loss_all:{}".format(mean_recon))


        for model_test in models:
            model_test.eval()
        data_id = int(tgt_id[-1])
        test_loss, test_score, pred_labels_DA, true_labels_DA ,_,_= evaluate_TGAT(shared_encoder, tgt_test_dl,criterion, config, device, data_id)
        test_loss_res.append(test_loss)

        print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
        if (test_loss < best_loss):
            best_loss = test_loss
            best_epoch = epoch
            best_loss_score = test_score
            checkpoint_best = {'model': shared_encoder,
                               'epoch': epoch,
                               'state_dict': shared_encoder.state_dict(),
                               'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_best,
                           f'D:/TGDSN/trained_models/best_model/{config["model_name"]}_{src_id}_{tgt_id}_best.pt')

        if (test_score < best_score):
            best_score = test_score
            best_score_epoch = epoch

    checkpoint_last = {'model': shared_encoder,
                               'epoch': epoch,
                               'state_dict': shared_encoder.state_dict(),
                               'optimizer': optimizer.state_dict()}
    torch.save(checkpoint_last,
                           f'D:/TGDSN/trained_models/last_model/{config["model_name"]}_{src_id}_{tgt_id}_last.pt')

    print('best_loss :{}'.format(best_loss))
    print('done')
    print('lr:{},d:{},r:{},f:{}'.format(args.lr, args.lambda_d, args.lambda_r, args.lambda_f))

    return test_loss, test_score,best_loss, best_epoch, best_loss_score ,best_score, best_score_epoch,manual_seed  #test_loss, test_score,



def source_only_result(params, device, config, model, my_dataset, src_id, tgt_id, args):
    hyper = params[f'{src_id}_{tgt_id}']
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    batch_size = hyper['batch_size']


    manual_seed = random.randint(1, 10000)

    random.seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)

    data_id = int(tgt_id[-1])

    ''' Load data '''
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[int(tgt_id[-1]) - 1], batch_size=hyper['batch_size'])


    if src_id == 'FD001' and tgt_id == 'FD003':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_16.pt',
            map_location=torch.device('cpu'))
        source_model = TGAT(hidden_dim=16).to(device)
        source_model.load_state_dict(checkpoint['state_dict'])


    elif src_id == 'FD002' and tgt_id == 'FD003':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_16.pt',
            map_location=torch.device('cpu'))
        source_model = TGAT(hidden_dim=16).to(device)
        source_model.load_state_dict(checkpoint['state_dict'])


    elif src_id == 'FD004' and tgt_id == 'FD001':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_16.pt',
            map_location=torch.device('cpu'))
        source_model = TGAT(hidden_dim=16).to(device)
        source_model.load_state_dict(checkpoint['state_dict'])


    elif src_id == 'FD002' and tgt_id == 'FD004':
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_24.pt',
            map_location=torch.device('cpu'))
        source_model = TGAT(hidden_dim=24).to(device)
        source_model.load_state_dict(checkpoint['state_dict'])

    else:
        checkpoint = torch.load(
            f'D:/TGDSN/trained_models/{config["model_name"]}_{src_id}_32.pt',
            map_location=torch.device('cpu'))
        source_model = TGAT(hidden_dim=32).to(device)
        source_model.load_state_dict(checkpoint['state_dict'])

    criterion = RMSELoss()

    src_only_loss, src_only_score, src_only_label, _, _, _ = evaluate_TGAT(source_model, tgt_test_dl, criterion, config,
                                                                       device, data_id)

    return src_only_loss, src_only_score

