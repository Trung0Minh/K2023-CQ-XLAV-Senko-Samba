from evaluation import main
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--data_root', default='./data/', type=str, help='rgbdata path')
    parser.add_argument('--train_steps', default=150000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')

    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='./results/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='HKU-IS+DUTS-TE+DUT-OMRON+ECSSD+PASCAL-S')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods2', type=str, default='Mamba', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./results/evaluation_results', help='path for saving result.txt')

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

    main.evaluate(args)
