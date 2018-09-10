#!/usr/bin/bash
LOCAL_STORAGE=/drives/C/Users/BLAH/
REMOTE_STORAGE=/drives/E/BLAH/
REMOTE_STORAGE2=/drives/E/BLAH/
declare -a FOLDER_LIST=("BLAH" "BLAH" "BLAH" "BLAH")

##Just in case - backup E:
# rsync -avh --progress ${REMOTE_STORAGE} ${REMOTE_STORAGE2}

for i in "${FOLDER_LIST[@]}"; do
	echo "$i"
	##Best of both worlds - fill gaps on C: and E:
	# rsync -avhu --progress ${LOCAL_STORAGE}${i}/ ${REMOTE_STORAGE}${i}/
	# rsync -avhu --progress ${REMOTE_STORAGE}${i}/ ${LOCAL_STORAGE}${i}/
	##Perfect harmony - replicate C: on E:
	rsync -avh --delete --progress ${LOCAL_STORAGE}${i}/ ${REMOTE_STORAGE}${i}/
done