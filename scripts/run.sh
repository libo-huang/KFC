# !/bin/bash
# bash -i run.sh <dataset> <gpu-id> <num_epochs>

cd ..
eval $(conda shell.bash hook)
conda activate KFC
conda info | egrep "conda version|active environment"

if [ "$1" != "" ]; then
    echo "Running on dataset: $1"
else
    echo "No dataset has been assigned."
fi

if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

if [ "$3" != "" ]; then
    echo "Running with # epochs: $3"
else
    echo "No # epochs has been assigned."
fi

dataArray = ('mnist' 'fashion' 'svhn')

for SEED in 0 1 2 3 4 5 6 7 8 9
do 
  if [ "${dataArray[@]}"  =~ "${1}" ]; then
	python main.py -dataset $1 -gpu $2 -epoch $3 -method KFC -ACC -generate -fid -seed $SEED;
  elif [ "$1" = "cifar" ]; then
	python main_cifar10.py -dataset cifar10 -gpu $2 -epoch $3 -method KFC -fid -seed $SEED;
  elif [ "$1" = "lsun" ]; then
	python main_lsun.py -dataset lsun -gpu $2 -epoch $3 -method KFC -fid -seed $SEED;
  else
	echo "No dataset has been assigned."
done
	
