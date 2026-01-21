import os.path as osp
import os
from .evaluator import Eval_thread
from .dataloader import EvalDataset


def evaluate(args):

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir
    gt_dir = args.data_root

    test_paths = args.test_paths.split('+')
    
    threads = []

    for dataset_name in test_paths:
        # Predictions are in results/DatasetName
        pred_dir_all = osp.join(pred_dir, dataset_name)
        
        # GTs are in data/DatasetName/Imgs
        gt_dir_all = osp.join(gt_dir, dataset_name, 'Imgs')

        print(f"Evaluating {dataset_name}...")
        print(f"  Preds: {pred_dir_all}")
        print(f"  GTs:   {gt_dir_all}")
        
        if not os.path.exists(pred_dir_all):
            print(f"  [Warning] Prediction directory not found: {pred_dir_all}")
            continue
        if not os.path.exists(gt_dir_all):
             print(f"  [Warning] GT directory not found: {gt_dir_all}")
             continue

        loader = EvalDataset(pred_dir_all, gt_dir_all)
        # Using 'Samba' as method name or just empty string if not needed for logging
        method_name = args.methods2 
        thread = Eval_thread(loader, method_name, dataset_name, output_dir, cuda=True)
        threads.append(thread)

    for thread in threads:
        print(thread.run())