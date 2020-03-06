python BasicCNNprune2.py \
--dataset cifar10 \
--test-batch-size 256 \
--depth 4 \
--percent 0.7 \
--model ./baseline/vgg16-cifar10_withnorank/EB-70-13.pth.tar \
--save ./baseline/vgg16-cifar10_withnorank/pruned_70_13_70 \
--gpu_ids 0
