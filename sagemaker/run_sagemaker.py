from sagemaker.pytorch import PyTorch

if __name__ == '__main__':
    train_params = {
        'max_disp': 100,
        'epochs': 1,
        'batch_size': 2
    }
    pytorch_estimator = PyTorch(entry_point='train.py',
                                source_dir='./',
                                train_instance_type='local_gpu',
                                role='AmazonSageMaker-ExecutionRole-20200127T105641',
                                train_instance_count=1,
                                framework_version='1.5.0',
                                dependencies=['../dataloader', '../utils', '../models'])
    pytorch_estimator.fit('s3://autogpe-datasets/hdsm_small')