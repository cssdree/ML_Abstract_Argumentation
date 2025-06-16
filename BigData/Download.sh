#!/bin/bash

echo "Downloading ICCMA 2017 instance set A..."
if [ -f "A.tar.gz" ]
then
	echo "Already exists!"
else
	wget http://argumentationcompetition.org/2017/A.tar.gz
fi
echo "Extracting archive..."
tar xvzf A.tar.gz --wildcards "*.apx"
for i in {1..5}
do
	mv A/"$i"/* A/
	rm -rf A/"$i"
done
echo "Generating incomplete AFs (this will take a while)..."
mkdir -p A-inc
python3 generate_instances.py A A-inc
echo "Done!"
rm A.tar.gz
rm -rf A

echo "Downloading ICCMA 2017 instance set B..."
if [ -f "B.tar.gz" ]
then
	echo "Already exists!"
else
	wget http://argumentationcompetition.org/2017/B.tar.gz
fi
echo "Extracting archive..."
tar xvzf B.tar.gz --wildcards "*.apx"
for i in {1..5}
do
	mv B/"$i"/* B/
	rm -rf B/"$i"
done
echo "Generating incomplete AFs (this will take a while)..."
mkdir -p B-inc
python3 generate_instances.py B B-inc
echo "Done!"
rm B.tar.gz
rm -rf B
