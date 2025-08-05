# Find all the changes between two versions of HOOMD-blue

You can use these bash and Python 3 scripts to find all the files that have been changed between 2 versions of HOOMD-blue (or all files in any 2 directories, including subfolders), and then step through the files one-by-one to see what the changes are.

This can be useful if you want to understand what changes were made by another user, or for identifying which changes you need to copy into a new version.
<br>

## How to use these scripts:

1. Move to the find_changes directory (`cd find_changes`)
2. In the bash script: `0_find_diffs`, set the "mod_dir" and "basic_dir" variables to the filepaths for the two directories you want to compare
3. Run the `0_find_diffs` script (you can do this locally or on Explorer), which will generate two text files: 
	- "uniquefiles.txt" a list of all files unique to one of the directories (i.e. new files that have been added) *NOTE: this includes .git files and other files that may not be significant*
	- "changedfiles.txt" a list of all files that have had their content modified 
4. [FILES ADDED] Use the list of unique files to determine which new files have been added
5. Leave the find_changes directory (`cd ..`)
6. Run the bash script `3_check_numchanges` to create a text file that lists the name of each file that has been changed and the number of changes that were made to that file. This will give you an idea of how much work you have to do if you want to investigate these changes.
7. Run the bash script `4_check_diffs" to view a side-by-side comparison of each file that has been changed. You can choose which file to look at, and you will be given information about the file and the option to exit before you view all the changes.

For example:
```
$ bash 4_check_diffs 
There are 34 files to check
Which file number do you want to start with? (1, 2, 3...)
1
Starting with file #1: BoxResizeUpdater.cc (43 changes)
View detailed changes? (Yes / No)? n
Exited on file #1: BoxResizeUpdater.cc (43 changes)
```

After you have loaded the side-by-side comparison the script will give you the option to continue to the next file. If you choose to exit, it will tell you which file you were **about** to view, so you know where to pick up next time.

```
Now ready to check file #2: BoxResizeUpdater.h (9 changes)
(Answer Yes to continue / No to exit)
View detailed changes? (Yes / No)? N
Exited on file #2: BoxResizeUpdater.h (9 changes)
```

**NOTE: You should also keep track of your progress somewhere else.** For example, I recommend copying the text from numchanges.txt into an .md file, converting it into a bulleted list, and checking off the files you have studied and/or copied changes from as you go (you may also want to check modfiles.txt to find the pruned subdirectory information from the full filepath):

EXAMPLE:
* :ballot_box_with_check: BoxResizeUpdater.cc : 43
* :ballot_box_with_check: BoxResizeUpdater.h : 9
* :o: Communicator.cc : 152
* :o: Communicator.h : 18
* ...
* `md/`
	* :o: ComputeThermo.cc
	* :o: ComputeThermo.h
	* ...
