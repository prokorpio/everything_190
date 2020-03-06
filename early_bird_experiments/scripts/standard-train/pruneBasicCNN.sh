python BasicCNNprune2.py \
--dataset cifar10 \
--test-batch-size 64 \
--depth 4 \
--percent 0.5 \
--model ./baseline/vgg16-cifar10/EB-50-6.pth.tar \
--save ./baseline/vgg16-cifar10/pruned_50_6_0.5 \
--gpu_ids 0
