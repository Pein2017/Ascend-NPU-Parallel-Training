from typing import Optional, Union
from argparse import Namespace


def main_worker(gpu: Optional[Union[str, int]], ngpus_per_node: int,
                args: Namespace):
    global best_acc1  # 假设这是一个全局变量，用于跟踪最佳精度

    args.gpu = args.process_device_map[gpu]

    if args.gpu is not None:
        print("Use GPU/NPU: {} for training".format(args.gpu))

    # 初始化分布式训练环境
    init_distributed_training(args, ngpus_per_node, gpu)

    # 设置设备
    device = set_device(args)

    # 创建或加载模型
    model = load_or_create_model(args)
    model.to(device)

    # 调整batch_size和workers
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # 数据加载代码
    if args.dummy:
        print("=> Dummy data is used!")
        # 使用假数据来模拟CIFAR-10的数据结构和大小
        train_dataset = datasets.FakeData(50000, (3, 32, 32), 10,
                                          transforms.ToTensor())
        val_dataset = datasets.FakeData(10000, (3, 32, 32), 10,
                                        transforms.ToTensor())
    else:
        # CIFAR-10数据集的实际目录
        data_path = args.data  # 例如：'/home/HW/Pein/cifar10_data/cifar-10-batches-py'
        batch_size = args.batch_size

        # 获取数据加载器
        train_loader, val_loader, test_loader = get_dataloaders(
            data_path, args.batch_size, distributed=args.distributed)

    # 定义损失函数（标准）和优化器
    criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数也移到相应设备
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 如果使用自动混合精度（AMP）
    if args.amp:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale=args.loss_scale)

    # 如果使用分布式数据并行
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if args.device == 'gpu' else None,
            broadcast_buffers=False)
    else:
        if args.device == 'gpu':
            model = torch.nn.DataParallel(model, device_ids=None)

    # 可选地从检查点恢复
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # 加载检查点，确保将数据加载到正确的设备上
            checkpoint = torch.load(args.resume, map_location=device)

            # 恢复训练的起始epoch和最佳精度
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            # 加载模型和优化器的状态
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 如果使用AMP，加载AMP的状态
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            # 如果找不到检查点文件，打印警告信息
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 为了提高效率，启用cudnn的benchmark模式
    # 这会让cudnn自动寻找最适合当前配置的算法
    # （特别是对于固定输入大小的情况）
    cudnn.benchmark = True

    # 如果指定了评估，则执行验证过程并返回
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
