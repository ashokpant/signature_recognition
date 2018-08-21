j=1; #total file processed 
folders=$(ls -d */) #total folders in the directory
k=0; # folder index
for f in $folders
do
	i=1; #files inside folders
	cd $f
	echo "Directory : $f"
	for name in *.*; do
   		echo " 	$k:$i - Processing image $name"
		echo "  $f${name} $k" >> ../train.txt
    		
    		i=`expr $i + 1`
	done
	echo "$i files processed in directory $f"
	cd ..
	k=`expr $k + 1`
	j=`expr $i + $j`
	echo "$j files processed so far."
done
echo "Total $j files processed!"
