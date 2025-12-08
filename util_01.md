## Folder structure

```
verf:
|-DATA
|	YA
|	|-set_##
|		|-s###.zip
|
|-data
|	|-s###
|
|-utils
	|-python scripts
```

## algorithm

```
alogrithm:
#-> is a digit
## and ### are numbers
traverse through the folder `DATA\ya`
|	|->traverse through the folders `set_##`
|	|	|-> select s###.zip
|	|	|	|-> extract this file in the current folder with the same name s###
|	|	|	|-> go into this folder
|	|	|	|	|-> convert each "{file_name}.csv" file in the s### folder to a "{file_name}-raw_targets.csv" and locate it in the folder `verf\data\s###`
|	|	|	|-> delete the this folder `verf\DATA\ya\set_##\s###`

```


## Logging:

log file 1: save the name with the current timestamp
1. log when each s### when each file is unzipped with a time stamp
2. log when each .csv is converted to the raw_target csv file
3. log when an error occurs
4. log when the s### folder is delted
5. differnce in step 1 and step 4 timestemp will give you the duration of processing also


log file 2: save the name with the current timestamp_size.csv
- seprate csv log for seeing the orginal.csv file size and next column raw_target.csv file size