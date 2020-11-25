
prerequisite: ncnn on linux-x86
	https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86


$cp proposal.h proposal.cpp ~/ncnn/src/layer
$cp rfcn.cpp ~/ncnn/examples
$cp 1.jpg rfcn.param rfcn.bin ~/ncnn/build/examples

$cd ~/ncnn/build/examples
$make 

$./rfcn 1.jpg

