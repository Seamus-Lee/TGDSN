import sys
sys.path.append("..")

import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from train_eval import *

fix_randomness(8884)

from models import *

def pre_train(model, train_dl, test_dl, data_id, config, params):
    # criteierion

    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    test_result = 10000
    for epoch in range(params['pretrain_epoch']):
        start_time = time.time()

        # TGAT单域训练
        train_loss, train_score, train_pred, train_labels = train_TGAT(model, train_dl, optimizer, criterion, config,
                                                                       device, data_id)

        scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Score: {train_score:7.3f}')

        if epoch % 5 == 0:

            test_loss, test_score, _, _, _, _ = evaluate_TGAT(model, test_dl, criterion, config, device, data_id)
            print('=' * 89)
            print(f'\t  Performance on test set::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')


        if params['save']:
            if (epoch % 10 == 0) and (test_loss < test_result):
                test_result = test_loss
                checkpoint1 = {'model': model,
                               'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict()}
                torch.save(checkpoint1,
                           f'D:/TGDSN/trained_models/{config["model_name"]}_FD00{data_id}_32.pt')

    test_loss, test_score, _, _, _, _ = evaluate_TGAT(model, test_dl, criterion, config, device, data_id)
    print('=' * 89)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    return model

