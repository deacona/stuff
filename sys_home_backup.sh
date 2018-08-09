

# #Best of both worlds - fill gaps on C: and E:
# rsync -avhu --progress /drives/C/Users/BLAH/Downloads/ /drives/E/data01/Downloads/
# rsync -avhu --progress "/drives/C/Users/BLAH/Google Drive/" "/drives/E/data01/Google Drive/"
# rsync -avhu --progress /drives/C/Users/BLAH/Music/ /drives/E/data01/Music/
# rsync -avhu --progress /drives/E/data01/Downloads/ /drives/C/Users/BLAH/Downloads/
# rsync -avhu --progress "/drives/E/data01/Google Drive/" "/drives/C/Users/BLAH/Google Drive/"
# rsync -avhu --progress /drives/E/data01/Music/ /drives/C/Users/BLAH/Music/

# #Just in case - backup E:
# rsync -avh --progress /drives/E/data01/ /drives/E/data04/

#Perfect harmony - replicate C: on E:
rsync -avh --delete --progress /drives/C/Users/BLAH/Downloads/ /drives/E/data01/Downloads/
rsync -avh --delete --progress "/drives/C/Users/BLAH/Google Drive/" "/drives/E/data01/Google Drive/"
rsync -avh --delete --progress /drives/C/Users/BLAH/OneDrive/ /drives/E/data01/OneDrive/
rsync -avh --delete --progress /drives/C/Users/BLAH/Music/ /drives/E/data01/Music/
